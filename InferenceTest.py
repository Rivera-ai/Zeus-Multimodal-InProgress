import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from Zeus import LLMConfig, LLM, ViTEncoder, CrossAttention, ImageDecoder, MultimodalModel, SpecialTokens

# I recommend only using CPU right now, because there are still some bugs when using CUDA
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model Configuration
gpt_config = LLMConfig(
    block_size=2048,
    vocab_size=50304,
    n_layer=12,
    n_head=12,
    n_embd=768,
    dropout=0.0,
    bias=True
)

vit_config = {
    'image_size': 256,
    'patch_size': 16,
    'dim': 512,
    'depth': 12,
    'heads': 8,
    'mlp_dim': 1024,
    'channels': 3
}


model = MultimodalModel(gpt_config, vit_config).to(DEVICE)

# Load checkpoint, you can download it at huggingface (Clarification: it is only a test checkpoint, it has only been trained in 61 steps) link: https://huggingface.co/RiveraAI/Zeus-Multimodal-InProgress
checkpoint_path = '/teamspace/studios/this_studio/ZeusViT/out/ckpt.pt'
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

# Print the checkpoint keys to verify
print("Claves en el checkpoint:", checkpoint.keys())

# Load model state ignoring missing keys
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()

# Function to preprocess images
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((vit_config['image_size'], vit_config['image_size'])),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)
    return image

# Function to preprocess text
def preprocess_text(text, tokenizer):
    tokens = tokenizer.encode(text, allowed_special={'<Start Image>'})
    tokens = torch.tensor(tokens).unsqueeze(0).to(DEVICE)
    return tokens

# Function for text to text inference
def text_to_text(model, text, tokenizer, max_new_tokens=50, temperature=1.0, top_k=None):
    tokens = preprocess_text(text, tokenizer)
    generated_tokens = model.generate(tokens, max_new_tokens, temperature, top_k)
    generated_text = tokenizer.decode(generated_tokens[0].tolist())
    return generated_text

# Function for inference of text and image to text
def text_image_to_text(model, text, image_path, tokenizer, max_new_tokens=50, temperature=1.0, top_k=None):
    tokens = preprocess_text(text, tokenizer)
    image = preprocess_image(image_path)
    combined_latents = model(tokens, images=image, phase="multimodal")

    # Adjust the shape of combined_latents to be [batch_size, seq_len]
    combined_latents = combined_latents.squeeze(1)  # Delete dimension of size 1

    # Convert combined_latents to Long type
    combined_latents = combined_latents.long()

    generated_tokens = model.generate(combined_latents, max_new_tokens, temperature, top_k)
    generated_text = tokenizer.decode(generated_tokens[0].tolist())
    return generated_text

# Function for text to image inference (Still buggy and not working properly, still in development/fixing)
def text_to_image(model, text, tokenizer):
    tokens = preprocess_text(text, tokenizer)
    print(f"Tokens shape: {tokens.shape}")
    generated_image = model(tokens, generate_image=True)
    generated_image = generated_image.squeeze().cpu().numpy()
    generated_image = (generated_image * 255).astype(np.uint8)
    return Image.fromarray(generated_image)

# Tokenizador
tokenizer = SpecialTokens().tokenizer

# Example
text_input = "Este es un ejemplo de texto."
image_path = "/teamspace/studios/this_studio/human.jpeg"
text2im = "Este es un ejemplo de texto. <Start Image>"

# Text2text
generated_text = text_to_text(model, text_input, tokenizer)
print("Texto generado:", generated_text)

# Text and image to text
generated_text_image = text_image_to_text(model, text_input, image_path, tokenizer)
print("Texto generado con imagen:", generated_text_image)

# Text2img
#generated_image = text_to_image(model, text2im, tokenizer)
#generated_image.show()
