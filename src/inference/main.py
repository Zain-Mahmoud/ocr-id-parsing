from fastapi import FastAPI, File, UploadFile
from infer import IDExtractionPipeline
from PIL import Image, ImageOps
import io 

pipeline = IDExtractionPipeline()

app = FastAPI()

@app.post("/get_info/")
async def get_info(file: UploadFile):
    raw_img = await file.read()
    im = Image.open(io.BytesIO(raw_img))
    im = ImageOps.exif_transpose(im)
    response = pipeline.predict(im)
    return response