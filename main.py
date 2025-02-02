from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import shutil
from PIL import Image
from torchvision import transforms
import uvicorn
import os

from model_loader import load_model

# Initialize FastAPI app
app = FastAPI()

# CORS Middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(device)

# Ensure required directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Image transformation settings
image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/remove-background/")
async def remove_background(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = Image.open(file_path).convert("RGBA")
    transformed_image = transform_image(image).unsqueeze(0).to(device)

    # Predict and generate the mask
    with torch.no_grad():
        preds = model(transformed_image)[-1].sigmoid().cpu()
    
    # Convert prediction to a mask
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)

    # Apply mask to the original image
    image.putalpha(mask)

    # Save the output
    output_path = f"outputs/no_bg_{file.filename}"
    image.save(output_path)

    return FileResponse(output_path, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
