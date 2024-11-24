from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pythonProject.TransformerRequirementsExtractor import model

app = FastAPI()


# Endpoint to predict the number from an image
@app.post("/predict/")
async def predict_answer(text):
    try:
        prediction = model.predict(text=text)
        # Return the predicted digit in a JSON response
        return JSONResponse(content={"May_be_this:": int(prediction)})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Health check route
@app.get("/")
def read_root():
    return {"message": "API is working!"}
