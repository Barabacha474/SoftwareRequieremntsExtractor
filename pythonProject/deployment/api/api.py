from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from TransformerRequirementsExtractor import RequirementExtractorNN
import os

app = FastAPI()

model = None

MODEL_PATH = "D:/SoftwareRequirementsExtractor/pythonProject/best_model.pt"
DATASET_PATH = "D:/SoftwareRequirementsExtractor/pythonProject/long_task_descriptions_en.csv"

class TextInput(BaseModel):
    text: str

def initialize_model():
    global model

    model = RequirementExtractorNN()

    if os.path.exists(MODEL_PATH):
        print("Loading pre-trained model...")
        model.load_model(MODEL_PATH)
    else:
        print("No pre-trained model found. Starting training...")
        model.train_from_dataset(file_path=DATASET_PATH)
        model.save_model(MODEL_PATH)
        print("Model trained and saved.")

@app.post("/predict/")
async def predict_answer(input: TextInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not initialized.")

    try:
        prediction = model.predict(text=input.text)
        return JSONResponse(content={"predicted_requirements": prediction})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def read_root():
    return {"message": "API is working!"}

if __name__ == "__main__":
    initialize_model()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)