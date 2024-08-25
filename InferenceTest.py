import torch
from torch.nn import functional as F
from Zeus import MultimodalModel, LLMConfig, ViTEncoder, DEVICE
import tiktoken
from PIL import Image
import numpy as np

tokenizer = tiktoken.get_encoding("gpt2")

def load_checkpoint(model, checkpoint_path):
    """Carga los pesos del modelo desde un checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    print(f"Checkpoint {checkpoint_path} cargado exitosamente.")
    return model

def generate_text(model, prompt, max_new_tokens=50, temperature=1.0, top_k=50):
    """Genera texto basado en un prompt inicial."""
    idx = torch.tensor(prompt).unsqueeze(0).to(DEVICE)
    generated_text = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return generated_text.squeeze(0).tolist()

def generate_image_from_text(model, prompt, image_size=(256, 256), max_new_tokens=0):
    """Genera una imagen basada en un prompt inicial."""
    idx = torch.tensor(prompt).unsqueeze(0).to(DEVICE)
    
    # Asegúrate de pasar max_new_tokens
    generated_image = model.generate(idx, max_new_tokens=max_new_tokens, generate_image=True)
    
    # Normalizar la imagen generada
    generated_image = generated_image.squeeze(0).cpu().detach().numpy()
    generated_image = np.clip(generated_image, 0, 1)
    return generated_image


def save_image(image_array, file_path):
    """Guarda la imagen generada en el sistema de archivos."""
    if image_array.shape != (256, 256, 3):
        print(f"Warning: Image shape {image_array.shape} is not (256, 256, 3)")
    else:
        print("Image shape is correct")

    image_array = (image_array * 255).astype(np.uint8) 
    image = Image.fromarray(image_array)
    image.save(file_path)
    print(f"Imagen guardada en {file_path}")


def generate_text_and_image(model, prompt, max_new_tokens=50, temperature=1.0, top_k=50):
    """Genera texto e imagen simultáneamente a partir de un prompt inicial."""
    idx = torch.tensor(prompt).unsqueeze(0).to(DEVICE)
    generated_text = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    generated_image = model.generate(idx, max_new_tokens=0, generate_image=True)
    return generated_text.squeeze(0).tolist(), generated_image.squeeze(0).cpu().detach().numpy()

if __name__ == "__main__":
    # Configuración del modelo
    gpt_config = LLMConfig()
    vit_config = {
        "image_size": 256,
        "patch_size": 16,
        "dim": 512,
        "depth": 12,
        "heads": 8,
        "mlp_dim": 1024,
        "channels": 3
    }

    model = MultimodalModel(gpt_config, vit_config)

    # Ruta al checkpoint
    checkpoint_path = "/teamspace/studios/this_studio/ZeusViT/out/ckpt.pt"
    model = load_checkpoint(model, checkpoint_path)

    # Ejemplo de generación de texto a texto
    prompt1 = 'Esto es una prueba.'
    prompt_text = tokenizer.encode(prompt1)  
    generated_text = generate_text(model, prompt_text)
    generated_text = tokenizer.decode(generated_text)
    print("Generated Text:", generated_text)

    prompt2 = 'Wild cat'
    prompt_text2 = tokenizer.encode(prompt2)

    # Ejemplo de generación de imagen a partir de texto
    generated_image = generate_image_from_text(model, prompt_text2)
    save_image(generated_image, "generated_image.png")

    # Ejemplo de generación de texto e imagen
    generated_text, generated_image = generate_text_and_image(model, prompt_text)
    generated_text = tokenizer.decode(generated_text)
    print("Generated Text:", generated_text)
    save_image(generated_image, "generated_text_and_image.png")
