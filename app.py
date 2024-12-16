from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('loan_default_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize FastAPI app
app = FastAPI()

# Define the input data schema
class LoanInput(BaseModel):
    Age: int
    Income: int
    LoanAmount: int
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float
    Education: str  # Categorical
    EmploymentType: str  # Categorical
    MaritalStatus: str  # Categorical
    HasMortgage: str  # Categorical
    HasDependents: str  # Categorical
    LoanPurpose: str  # Categorical
    HasCoSigner: str  # Categorical

@app.get("/")
def read_root():
    return {"message": "Loan Default Prediction API"}

@app.post("/predict")
def predict(input_data: LoanInput):
    try:
        # Map categorical features to numerical values based on the training dataset encoding
        categorical_mappings = {
            "Education": {"High School": 0, "Bachelor's": 1, "Master's": 2},
            "EmploymentType": {"Full-time": 0, "Part-time": 1, "Unemployed": 2},
            "MaritalStatus": {"Single": 0, "Married": 1, "Divorced": 2},
            "HasMortgage": {"No": 0, "Yes": 1},
            "HasDependents": {"No": 0, "Yes": 1},
            "LoanPurpose": {"Auto": 0, "Business": 1, "Other": 2},
            "HasCoSigner": {"No": 0, "Yes": 1},
        }

        # Convert input data to a numerical format
        input_array = np.array([[
            input_data.Age,
            input_data.Income,
            input_data.LoanAmount,
            input_data.CreditScore,
            input_data.MonthsEmployed,
            input_data.NumCreditLines,
            input_data.InterestRate,
            input_data.LoanTerm,
            input_data.DTIRatio,
            categorical_mappings["Education"].get(input_data.Education, -1),
            categorical_mappings["EmploymentType"].get(input_data.EmploymentType, -1),
            categorical_mappings["MaritalStatus"].get(input_data.MaritalStatus, -1),
            categorical_mappings["HasMortgage"].get(input_data.HasMortgage, -1),
            categorical_mappings["HasDependents"].get(input_data.HasDependents, -1),
            categorical_mappings["LoanPurpose"].get(input_data.LoanPurpose, -1),
            categorical_mappings["HasCoSigner"].get(input_data.HasCoSigner, -1),
        ]])

        # Check for invalid categorical values
        if -1 in input_array:
            raise ValueError("Invalid categorical value provided in input data.")

        # Scale the input data
        scaled_data = scaler.transform(input_array)

        # Make prediction
        probabilities = model.predict_proba(scaled_data)[0]
        prediction = model.predict(scaled_data)[0]

        # Return prediction and probabilities
        return {
            "prediction": int(prediction),  # 0: No Default, 1: Default
            "probability": {
                "no_default": probabilities[0],
                "default": probabilities[1]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
