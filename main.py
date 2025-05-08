import pandas as pd
from flask import Flask, request, render_template
import pickle
import numpy as np
from collections import defaultdict

app = Flask(__name__)

# Load model and data
model = pickle.load(open("churn_model.pkl", "rb"))
data = pd.read_csv("new_data_set4.csv")

# Store predictions (in-memory)
predictions = defaultdict(dict)

def predict_churn_for_all():
    """Predict churn for all numbers in the dataset"""
    predictions.clear()
    for _, row in data.iterrows():
        # Convert to string and remove .0 if present
        phone_number = str(row['mobile_no']).replace('.0', '') if '.0' in str(row['mobile_no']) else str(row['mobile_no'])
        try:
            input_data = row.drop('mobile_no').values.astype(np.float32).reshape(1, -1)
            churn_prob = model.predict(input_data)[0][0]
            churn_percent = round(float(churn_prob) * 100, 2)

            # Categorize
            if churn_percent >= 70:
                category = "high"
            elif churn_percent >= 40:
                category = "medium"
            else:
                category = "low"

            predictions[phone_number] = {
                "percentage": churn_percent,
                "category": category
            }
        except Exception as e:
            print(f"Error predicting for {phone_number}: {str(e)}")

# Predict all numbers when starting the app
predict_churn_for_all()

@app.route("/")
def home():
    return render_template("index1.html", predictions=predictions)

@app.route("/predict", methods=["POST"])
def predict_single():
    try:
        phone_number = request.form["phoneNumber"]

        # Validate input and remove .0 if present
        try:
            phone_number = phone_number.replace('.0', '') if '.0' in phone_number else phone_number
            phone_int = int(float(phone_number))  # Handle cases where number might come as float string
            phone_number = str(phone_int)  # Convert back to string without .0
        except ValueError:
            return render_template("index1.html",
                                 predictions=predictions,
                                 error="Invalid phone number format")

        # Check if number exists (compare as float to match dataset)
        record = data[data["mobile_no"].astype(str).str.replace('.0', '') == phone_number]
        if record.empty:
            return render_template("index1.html",
                                 predictions=predictions,
                                 error=f"Number {phone_number} not found")

        # Predict
        input_data = record.drop(columns=["mobile_no"]).values.astype(np.float32)
        churn_prob = model.predict(input_data)[0][0]
        churn_percent = round(float(churn_prob) * 100, 2)

        # Categorize
        if churn_percent >= 70:
            category = "high"
        elif churn_percent >= 40:
            category = "medium"
        else:
            category = "low"

        # Update predictions
        predictions[phone_number] = {
            "percentage": churn_percent,
            "category": category
        }

        return render_template("index1.html",
                             predictions=predictions,
                             success=f"Updated prediction for {phone_number}")

    except Exception as e:
        return render_template("index1.html",
                             predictions=predictions,
                             error=f"System error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)