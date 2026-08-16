from fastapi import FastAPI, File, UploadFile
from infer import predict, load
from PIL import Image, ImageOps
import io 

models = load()
yolo = models["yolo"]
vlm_generation, vlm_tokenizer = models["vlm"]
digit_model = models["digit_model"]
ocr_reader = models["ocr_reader"]

app = FastAPI()

@app.post("/get_info/")
async def get_info(file: UploadFile):
    raw_img = await file.read()
    im = Image.open(io.BytesIO(raw_img))
    im = ImageOps.exif_transpose(im)
    response = predict(yolo, vlm_generation, vlm_tokenizer, digit_model, ocr_reader, im)
    return response