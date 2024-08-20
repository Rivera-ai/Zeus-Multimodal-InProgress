import os
import time
import math
import pickle
from contextlib import nullcontext
import argparse

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from Zeus import SpecialTokens, LLM, LLMConfig, MultimodalModel
import pandas as pd
from PIL import Image
import csv
import random
import tiktoken
import torchvision.transforms as transforms

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 61
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'train'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 2048

data_files = ['train_es.bin', 'trainbookcorpus.bin', 'trainopenwebtext.bin']
last_indices = {file: 0 for file in data_files}
last_tokens = {file: None for file in data_files}
file_steps = 10
file_idx = 0
file_step_counter = 0

text_phase_iters = 30  # Número de iteraciones para entrenar solo con texto
text_image_phase_iters = 30  # Número de iteraciones para entrenar con texto e imagen
multimodal_phase = False  # Bandera para saber si estamos en la fase multimodal
phase_iter_count = 0


# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

gpt_config = LLMConfig(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=50304,
    dropout=dropout
)

# Configuración del ViT (Encoder de Imagen)
vit_config = {
    'image_size': 256,
    'patch_size': 16,
    'dim': 512,
    'depth': 12,
    'heads': 8,
    'mlp_dim': 1024,
    'channels': 3
}

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Inicializar SpecialTokens y obtener el tokenizador extendido
special_tokens = SpecialTokens()
tokenizer = special_tokens.tokenizer  # Usa el tokenizador extendido

# poor man's data loader
data_dir = os.path.join('datasettexto', dataset)
def get_batch(split, file_idx):
    data_path = os.path.join(data_dir, data_files[file_idx]) if split == 'train' else os.path.join(data_dir, 'val.bin')
    print(f"Using dataset: {data_files[file_idx]}")  # Imprimir el dataset actual
    
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    
    if split == 'train':
        start_idx = last_indices[data_files[file_idx]]
        end_idx = start_idx + block_size * batch_size
        if end_idx >= len(data):
            start_idx = 0
            end_idx = block_size * batch_size

        ix = torch.arange(start_idx, end_idx, block_size)
        
        # Actualizar los índices y los tokens
        last_indices[data_files[file_idx]] = end_idx
        last_tokens[data_files[file_idx]] = (ix, data[ix])
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        #print(f"Batch shapes - X: {x.shape}, Y: {y.shape}")
        
        # Imprimir los índices y tokens actuales
        #print(f"start_idx: {start_idx}, end_idx: {end_idx}")
        #print(f"indices: {ix}")
        #print(f"tokens: {data[ix]}")
    else:
        ix = torch.randint(len(data) - block_size, (batch_size,))
    
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    #print(f"X: {x}")
    #print(f"Y: {y}")
    
    return x, y

def load_flickr30k(csv_path, img_dir):
    data = pd.read_csv(csv_path)
    pairs = []
    for idx, row in data.iterrows():
        descriptions = eval(row['raw'])  # Convertir el string a lista
        img_path = os.path.join(img_dir, row['filename'])
        for desc in descriptions:
            pairs.append((img_path, desc))
    return pairs

def load_coco(csv_path):
    data = pd.read_csv(csv_path)
    pairs = []
    base_path = '/teamspace/studios/this_studio/COCO/'
    for idx, row in data.iterrows():
        descriptions = eval(row['descriptions'])  # Convertir el string a lista
        img_path = os.path.join(base_path, row['image_path'])
        for desc in descriptions:
            pairs.append((img_path, desc))
    return pairs


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    
    model = MultimodalModel(gpt_config=gpt_config, vit_config=vit_config).to(device)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = MultimodalModel(gpt_config=gpt_config, vit_config=vit_config).to(device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    # Actualizar índices y demás información desde el checkpoint
    last_indices = checkpoint.get('last_indices', last_indices)
    image_pair_index = checkpoint.get('last_image_pair_index', 0)
    cycle_num = checkpoint.get('last_cycle_num', 0)

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = LLM.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
#if block_size < model.config.block_size:
#    model.crop_block_size(block_size)
#    model_args['block_size'] = block_size # so that the checkpoint will have the right value
#model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, file_idx)
            loss = None
            #print(f"Batch shapes - X: {X.shape}, Y: {Y.shape}")

            with ctx:
                if multimodal_phase:
                    #print(f"Evaluando en Modalidad - Multimodal (Texto + Imagen) en iteración {k}")
                    
                    # Verifica si realmente tenemos una imagen
                    if k < len(all_image_pairs):
                        img_path, description = all_image_pairs[k]
                        try:
                            image = Image.open(img_path).convert("RGB")
                            image = transform(image).unsqueeze(0).to(device)
                            #print(f"Image shape: {image.shape}")
                        except Exception as e:
                            print(f"Error loading image: {str(e)}")
                            image = None
                    else:
                        image = None

                    # Si hay imagen, aplica atención cruzada, si no, trata como solo texto
                    if image is not None:
                        combined_latents = model(X, images=image, phase="multimodal")
                        #print(f"combined_latents shape: {combined_latents.shape}")
                         # Imprimir estadísticas como media y desviación estándar
                        print(f"combined_latents mean: {combined_latents.mean().item():.4f}")
                        print(f"combined_latents std: {combined_latents.std().item():.4f}")
                        print(f"combined_latents first few elements: {combined_latents.view(-1)[:5]}")
                        #logits = model.gpt.lm_head(logits)
                        #loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
                    else:
                        logits, loss = model(X, targets=Y, phase="text")
                        #print(f"Logits shape: {logits.shape}, Targets shape: {Y.shape}")
                    
                else:
                    #print(f"Evaluando en Modalidad - Solo Texto en iteración {k}")
                    logits, loss = model(X, targets=Y, phase="text")
                    #print(f"Logits shape: {logits.shape}, Targets shape: {Y.shape}")

            if loss is not None:
                losses[k] = loss.item()
            else:
                print(f"Warning: No se pudo calcular la pérdida en la iteración {k}, conjunto {split}.")
                
        out[split] = losses.mean()
    model.train()
    return out







# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# training loop
X, Y = get_batch('train', file_idx)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
# Inicializar variables para el entrenamiento multimodal
if multimodal_phase:
    # Shuffle para asegurar que se entrenen diferentes combinaciones de texto e imagen
    random.shuffle(all_image_pairs)
    image_pair_index = 0


while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Alternar entre fases de entrenamiento
    if not multimodal_phase and phase_iter_count >= text_phase_iters:
        # Hemos completado una fase de texto, cambiamos a la fase multimodal
        multimodal_phase = True
        phase_iter_count = 0  # Reiniciar contador de iteraciones
        print("Cambiando a la fase de entrenamiento multimodal (texto + imagen)")
        torch.cuda.empty_cache()  # Limpiar memoria de datasets de texto
        image_pair_index = 0

    elif multimodal_phase and phase_iter_count >= text_image_phase_iters:
        # Hemos completado una fase multimodal, volvemos a la fase de texto
        multimodal_phase = False
        phase_iter_count = 0  # Reiniciar contador de iteraciones
        print("Cambiando a la fase de entrenamiento solo texto")
        torch.cuda.empty_cache()  # Limpiar memoria de datasets multimodales
        file_idx = 0  # Reiniciar índice del dataset de texto si es necesario

    # Obtener el siguiente lote de datos
    if multimodal_phase:

        # Cargar datasets de texto e imagen
        flickr_pairs = load_flickr30k('/teamspace/studios/this_studio/flickr30k/flickr_annotations_30k.csv', '/teamspace/studios/this_studio/flickr30k/flickr30k-images/')
        coco_pairs = load_coco('/teamspace/studios/this_studio/COCO/COCO_DATASET.csv')
        all_image_pairs = flickr_pairs + coco_pairs
        random.shuffle(all_image_pairs)
        # Entrenamiento multimodal: Seleccionar una imagen y su descripción
        img_path, description = all_image_pairs[image_pair_index]
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)  # Esta línea estaba faltando
        
        # Tokenizar la descripción
        tokens = tokenizer.encode(description)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

        # Obtener los objetivos
        X, Y = tokens, tokens

        # Avanzar al siguiente par
        image_pair_index += 1
        if image_pair_index >= len(all_image_pairs):
            image_pair_index = 0
            random.shuffle(all_image_pairs)
    else:
        # Fase de solo texto
        X, Y = get_batch('train', file_idx)

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
            })
        print(f"Evaluating whether to save checkpoint at step {iter_num}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            print(f"Saving checkpoint at step {iter_num} with val loss {losses['val']:.4f}")
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'last_indices': last_indices,
                    'last_image_pair_index': image_pair_index,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                print(f"Checkpoint saved at step {iter_num}")

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            if multimodal_phase:
               #print(f"Image shape before passing to model: {image.shape}") 
               combined_latents = model(X, images=image, phase="multimodal")
               print(f"combined_latents shape: {combined_latents.shape}")
               # Imprimir estadísticas como media y desviación estándar
               print(f"combined_latents mean: {combined_latents.mean().item():.4f}")
               print(f"combined_latents std: {combined_latents.std().item():.4f}")
               print(f"combined_latents first few elements: {combined_latents.view(-1)[:5]}")
               #logits = model.gpt.lm_head(logits)  # Aplica la proyección final
               #loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
            else:
                logits, loss = model(X, targets=Y, phase="text")
                loss = loss / gradient_accumulation_steps
            #X, Y = get_batch('train')
                #print(f"Micro Step {micro_step}: Logits shape: {logits.shape}, Targets shape: {Y.shape}, Loss: {loss}")


            if loss is None:
                #print(f"Warning: Loss is None at iteration {iter_num}, micro_step {micro_step}")
                #print(f"Logits shape: {logits.shape}")
                #print(f"Expected target shape: {Y.view(-1).shape}")
                continue
            loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
            

        #scaler.scale(loss).backward()
        X, Y = get_batch('train', file_idx)

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
    iter_num += 1
    phase_iter_count += 1
    local_iter_num += 1
    file_step_counter += 1
    if file_step_counter >= file_steps:
        file_step_counter = 0
        file_idx = (file_idx + 1) % len(data_files)
        print(f"Switching to dataset: {data_files[file_idx]} with last index: {last_indices[data_files[file_idx]]}")
        X, Y = get_batch('train', file_idx)

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()