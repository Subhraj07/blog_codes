from fastapi import FastAPI
import uvicorn
from torchvision import models
from predict import predicted_results

app = FastAPI()

img_path = "dog.jpg"

pymodel = models.resnet101(pretrained=True)

# query parameter http://127.0.0.1:8000/predict/?img_path=0
@app.get("/predict/")
async def predict(img_path: str = ""):
    return predicted_results(img_path, pymodel)

@app.post("/update_model")
async def read_item():
    global pymodel
    pymodel = models.alexnet(pretrained=True)
    return {"message": "model updated sucessfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)