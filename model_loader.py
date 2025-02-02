import torch
from transformers import AutoModelForImageSegmentation

def load_model(device):
    model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True)
    model.to(device)
    model.eval()
    return model
