from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['file']
    data = pd.read_csv(file)

    # Clean column names
    data.columns = data.columns.str.strip()

    # Debug: print incoming columns
    print("Uploaded file columns:", data.columns.tolist())

    required_columns = ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']

    # Check if all required columns are present
    if not all(col in data.columns for col in required_columns):
        missing = [col for col in required_columns if col not in data.columns]
        return f"Error: Missing columns in uploaded file: {missing}"

    # Make prediction
    predictions = model.predict(data[required_columns])
    data['Prediction'] = predictions

    return render_template("index.html", predictions=data.to_html(classes='data'))


if __name__ == "__main__":
    app.run(debug=True)
