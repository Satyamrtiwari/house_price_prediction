from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load("house_price_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")  # Serve HTML file

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Get JSON data from frontend
    features = np.array([[data["Size"], data["Bedrooms"], data["Age"]]])  # Convert to array
    prediction = model.predict(features)[0]  # Predict price

    return jsonify({"Predicted Price": f"${prediction:,.2f}"})  # Send response

if __name__ == "__main__":
    app.run(debug=True)
