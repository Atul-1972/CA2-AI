
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# Create FastAPI application
app = FastAPI(title="CA2-AI Wine Classification API")

# Load trained model
model_path = Path(__file__).parent / "model.joblib"

data = joblib.load(model_path)
model = data["model"]
features = data["features"]


# Define input features
class WineInput(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


# Home endpoint
@app.get("/")
def home():
    return {"message": "Wine Classification API Running"}


# Health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}


# Prediction endpoint
@app.post("/predict")
def predict(wine: WineInput):

    # Convert input into dictionary
    values = wine.model_dump()

    # Fix feature name mismatch
    values["od280/od315_of_diluted_wines"] = values.pop(
        "od280_od315_of_diluted_wines"
    )

    # Create DataFrame with correct feature names
    input_data = pd.DataFrame(
        [[values[name] for name in features]],
        columns=features
    )

    # Predict wine class
    prediction = model.predict(input_data)

    return {
        "predicted_class": int(prediction[0])
    }
