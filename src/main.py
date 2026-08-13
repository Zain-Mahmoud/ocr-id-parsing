from fastapi import FastAPI, File, UploadFile
from inference.infer import predict, load
from PIL import Image

models = load()
yolo = models["yolo"]
vlm_generation, vlm_tokenizer = models["vlm"]
ocr_model = models["ocr"]

app = FastAPI()

@app.post("/predict_image/")
async def predict_image(file: UploadFile):
    try:
        im = Image.open(file.file)
    except:
        return {"Error": 404}
    response = predict(yolo, vlm_generation, vlm_tokenizer, ocr_model, im)
    return response

