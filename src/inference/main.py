from fastapi import FastAPI, File, UploadFile
from .infer import predict, load
from PIL import Image, ImageOps
import io 

models = load()
yolo = models["yolo"]
vlm_generation, vlm_tokenizer = models["vlm"]
ocr_model = models["ocr"]

app = FastAPI()

@app.post("/predict_image/")
async def predict_image(file: UploadFile):
    raw_img = await file.read()
    im = Image.open(io.BytesIO(raw_img))
    im = ImageOps.exif_transpose(im)
    response = predict(yolo, vlm_generation, vlm_tokenizer, ocr_model, im)
    return response

