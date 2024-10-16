import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from einops import rearrange, repeat

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

DEVICE = get_device()

class LayerNorm(nn.Module):
    """LayerNorm pero con un sesgo opcional. PyTorch no admite simplemente bias = Falso"""
    def __init__(self, ndim, bias) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttetion(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0
        # key, query, value proyecciones de valor para todas las cabezas pero en un lote
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # salida de proyección
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularización

        self.att_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attetion')
        if not self.flash:
            print("En caso de que salga esto en la terminal actualiza PyTorch a la versión más reciente")

            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # el batch size, sequence length, embedding dimensionalidad de incrustación del (n_embd)

        # calcular la query, la key, y los values para todas las cabezas de lote (batch) y mover el cabezal hacia delante para que el lote esté atenuado

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, nh, T, hs)

        if self.flash:
            # Eficiencia de attetion usando Flash Attetion con los CUDA Kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)

        else:
            # Implementación manual de la Attetion

            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.att_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Salida de la proyección(output projection)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
        """clase MLP (Multi-Layer Perceptron) es una parte fundamental del modelo GPT. Se utiliza para procesar las representaciones de los datos entre capas de atención. """

        def __init__(self, config):
            super().__init__()

            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.gelu = nn.GELU()
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x):
            x = self.c_fc(x)
            x = self.gelu(x)
            x = self.c_proj(x)
            x = self.dropout(x)
            return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttetion(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, image_size=256, patch_size=16, dim=512, depth=12, heads=8, mlp_dim=1024, channels=3):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, heads, mlp_dim, batch_first=True),
            num_layers=depth
        )

    def forward(self, img):
        p = self.patch_size
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, heads, num_patches):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads)
        self.num_patches = num_patches

    def forward(self, text_latents, image_latents):
        print(f"Text latents input shape: {text_latents.shape}")
        print(f"Image latents input shape: {image_latents.shape}")
        # Transponer los tensores para que tengan la forma esperada por nn.MultiheadAttention
        text_latents = text_latents.transpose(0, 1)
        image_latents = image_latents.transpose(0, 1)
        
        # Aplicar la atención cruzada
        attn_output, _ = self.attn(text_latents, image_latents, image_latents)
        print(f"Attention output shape: {attn_output.shape}")
        
        # Volver a transponer el resultado para que coincida con la forma original
        attn_output = attn_output.transpose(0, 1)

        # Ajustar el número de "latents" para igualar num_patches
        if attn_output.size(1) < self.num_patches:
            attn_output = F.pad(attn_output, (0, 0, 0, self.num_patches - attn_output.size(1)))
        elif attn_output.size(1) > self.num_patches:
            attn_output = attn_output[:, :self.num_patches, :]
        
        print(f"Combined latents output shape after padding/truncation: {attn_output.shape}")
        
        return attn_output



class ImageDecoder(nn.Module):
    def __init__(self, dim=512, patch_size=16, image_size=256, channels=3):
        super().__init__()
        self.latent_projection = nn.Linear(dim, dim)
        self.latent_to_patch = nn.Linear(dim, patch_size**2 * channels)
        self.patch_size = patch_size
        self.image_size = image_size
        self.channels = channels

    def forward(self, latents):
        # Verificar las dimensiones de los latentes
        print(f"Latents input shape: {latents.shape}")
        num_patches = (self.image_size // self.patch_size) ** 2
        assert latents.size(1) == num_patches, \
            f"Expected {num_patches} latents, but got {latents.size(1)}"
        print(f"Latents shape: {latents.shape}")

        latents = self.latent_projection(latents)
        patches = self.latent_to_patch(latents)
        print(f"Patches shape after latent_to_patch: {patches.shape}")

        # Revisar las dimensiones de los patches
        expected_patch_dim = self.patch_size ** 2 * self.channels
        assert patches.size(-1) == expected_patch_dim, \
            f"Expected patch dimension {expected_patch_dim}, but got {patches.size(-1)}"

        # Reconstrucción de la imagen desde los patches
        img = rearrange(
            patches, 
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', 
            h=self.image_size // self.patch_size, 
            w=self.image_size // self.patch_size, 
            p1=self.patch_size, 
            p2=self.patch_size, 
            c=self.channels
        )
        print(f"Reconstructed image shape: {img.shape}")

        return img




@dataclass

class LLMConfig:
    block_size: int = 2048
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class LLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.to(DEVICE)

        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Con peso condicionado cuando se usa torch.compile() se generan algunas advertencias
        #  "UserWarning: functional_call was passed multiple values for tied weights.
        # Este comportamiento está obsoleto y será un error en versiones futuras
        # No estoy 100% seguro de qué es esto, hasta ahora parece inofensivo

        self.transformer.wte.weight = self.lm_head.weight

        #init de los weights

        self.apply(self._init_weights)

        # Inicio escalado espacial a las proyecciones residuales, basado en GPT-2
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Reportar el numero de parametros
        print("Numero de parametros: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):

        """Devuelve el número de parámetros del modelo.
            Para el recuento sin incrustaciones (predeterminado), las incrustaciones de posición se restan.
            Las incrustaciones de tokens también lo harían, excepto debido al parámetro que comparte estos
            Los parámetros en realidad se usan como pesos en la capa final, por lo que los incluimos."""

        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_latents=False):
        device = idx.device
        b, t = idx.size()

        assert t <= self.config.block_size, f"No se puede reenviar una secuencia de longitud {t}, el tamaño del bloque es solo {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Forward del modelo LLM (GPT) en sí

        tok_emb = self.transformer.wte(idx) # incrustaciones de tokens de forma (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # posicionar incrustaciones de forma (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block  in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if return_latents:
            return x

        if targets is not None:
            # Si nos dan algunos objetivos deseados, también calculamos la pérdida.
            logits = self.lm_head(x)
            #print(f"Logits shape: {logits.shape}, Targets shape: {targets.shape}")
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Minioptimización del tiempo de inferencia: solo reenvía lm_head en la última posición
            logits = self.lm_head(x[:, [-1], :]) # nota: usar la lista [-1] para preservar el tiempo atenuado
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # En caso que tengas que disminuir el tamaño del bloque solo si es necesario
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Comenzar con todos los parámetros candidatos
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filtrar aquellos que no requieren graduación (grad)
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Definir los grups de optimizadores. Cualquier parámetro que sea 2D perderá peso; de lo contrario, no.
        # Es decir, todos los tensores de peso en matmuls + embeddings decaen, todos los sesgos y normas de capas no lo hacen.

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups =[
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Numero de parametros que decaen: {len(decay_params)}, de {num_decay_params:,} parametros")
        print(f"Numero de parametros que no decaen: {len(nodecay_params)}, de {num_nodecay_params:,} parametros")

        # Creando el optimizador AdamW y usando la versión fusionada si esta disponible
        fused_avaible = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        used_fused = fused_avaible and device_type == 'cuda'
        extra_args = dict(fused=True) if used_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Usando el fused de AdamW: {used_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of L4 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 242e12 # NVIDIA L4 GPU FP16 Tensor Core peak flops is 242 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu


class MultimodalModel(nn.Module):
    def __init__(self, gpt_config, vit_config):
        super().__init__()
        self.to(DEVICE)
        self.gpt = LLM(gpt_config)
        self.vit = ViTEncoder(**vit_config)
        self.cross_attn = CrossAttention(dim=vit_config['dim'], heads=vit_config['heads'], num_patches=(vit_config['image_size'] // vit_config['patch_size']) ** 2)
        self.decoder = ImageDecoder(dim=vit_config['dim'])
        self.text_projection = nn.Linear(gpt_config.n_embd, vit_config['dim'])
        self.projection = nn.Linear(512, 768)
        
        #self.lm_head = self.gpt.lm_head

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.gpt.config.vocab_size, self.gpt.config.n_embd),
            wpe = nn.Embedding(self.gpt.config.block_size, self.gpt.config.n_embd),
            drop = nn.Dropout(self.gpt.config.dropout),
            h = nn.ModuleList([Block(self.gpt.config) for _ in range(self.gpt.config.n_layer)]),
            ln_f = LayerNorm(self.gpt.config.n_embd, bias=self.gpt.config.bias)
        ))

        

    def forward(self, idx, images=None, targets=None, generate_image=False, phase="text"):
        if phase == "multimodal" and images is not None:
            # Modalidad texto+imagen
            image_latents = self.vit(images)[:, 1:]  # Eliminar el token CLS
            text_latents = self.gpt(idx, return_latents=True)
            text_latents = self.text_projection(text_latents)
            combined_latents = self.cross_attn(text_latents, image_latents)
            print(f"Combined latents shape: {combined_latents.shape}")
            
            if generate_image:
                # Decodificar la imagen
                return self.decoder(combined_latents)
                
            else:
                # Predecir el siguiente token (texto)
                projected_latents = self.projection(combined_latents)
                logits = self.gpt.lm_head(projected_latents[:, -1, :])
                loss = None
                if targets is not None:
                    logits = logits.view(-1, logits.size(-1))  # (N * T, vocab_size)
                    targets = targets.view(-1)
                    loss = F.cross_entropy(logits, targets, ignore_index=-1)
                    print(f"logits shape: {logits.shape}, targets shape: {targets.shape}")
                return logits, loss
        else:
            # Modalidad solo texto
            logits, loss = self.gpt(idx, targets=targets)
            return logits, loss


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Comenzar con todos los parámetros candidatos
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filtrar aquellos que no requieren graduación (grad)
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Definir los grups de optimizadores. Cualquier parámetro que sea 2D perderá peso; de lo contrario, no.
        # Es decir, todos los tensores de peso en matmuls + embeddings decaen, todos los sesgos y normas de capas no lo hacen.

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups =[
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Numero de parametros que decaen: {len(decay_params)}, de {num_decay_params:,} parametros")
        print(f"Numero de parametros que no decaen: {len(nodecay_params)}, de {num_nodecay_params:,} parametros")

        # Creando el optimizador AdamW y usando la versión fusionada si esta disponible
        fused_avaible = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        used_fused = fused_avaible and device_type == 'cuda'
        extra_args = dict(fused=True) if used_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Usando el fused de AdamW: {used_fused}")

        return optimizer

    def get_num_params(self, non_embedding=True):

        """Devuelve el número de parámetros del modelo.
            Para el recuento sin incrustaciones (predeterminado), las incrustaciones de posición se restan.
            Las incrustaciones de tokens también lo harían, excepto debido al parámetro que comparte estos
            Los parámetros en realidad se usan como pesos en la capa final, por lo que los incluimos."""

        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.gpt.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of L4 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 242e12 # NVIDIA L4 GPU FP16 Tensor Core peak flops is 242 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, images=None, temperature=1.0, top_k=None, generate_image=False):
        print("Entering generate method")
        if  generate_image and images is not None:
            print("Generating image...")
            # Generar imagen condicionada en el texto
            image_latents = self.vit(images)[:, 1:]
            print(f"Image latents shape after ViT: {image_latents.shape}")
            text_latents = self.gpt(idx, return_latents=True)
            print(f"Text latents shape from GPT: {text_latents.shape}")
            text_latents = self.text_projection(text_latents)
            print(f"Text latents shape after projection: {text_latents.shape}")
            combined_latents = self.cross_attn(text_latents, image_latents)
            print(f"Combined latents shape after CrossAttention: {combined_latents.shape}")
            num_patches = (self.decoder.image_size // self.decoder.patch_size) ** 2
            assert combined_latents.size(1) == num_patches, \
                f"Expected {num_patches} combined latents, but got {combined_latents.size(1)}"

            return self.decoder(combined_latents)
        
        else:
            
            generated_tokens = idx.clone()

            for _ in range(max_new_tokens):
            # Ajusta la longitud del contexto según el tamaño del bloque
                idx_cond = idx if idx.size(1) <= self.gpt.config.block_size else idx[:, -self.gpt.config.block_size:]

            # Paso forward en el modelo para obtener los logits
                logits, _ = self.gpt(idx_cond)

            # Escalar logits según la temperatura
                logits = logits[:, -1, :] / temperature

            # Recorte los logits al top_k más alto
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

            # Convertir logits a probabilidades
                probs = F.softmax(logits, dim=-1)

            # Muestreo de las probabilidades
                idx_next = torch.multinomial(probs, num_samples=1)

            # Agregar el índice de muestra a la secuencia en ejecución
                idx = torch.cat((idx, idx_next), dim=1)

                generated_tokens = torch.cat((generated_tokens, idx_next), dim=1)


            return idx
