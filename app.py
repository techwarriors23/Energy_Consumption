from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load models
model_zone1 = joblib.load("rf_zone1.pkl")
model_zone2 = joblib.load("rf_zone2.pkl")
model_zone3 = joblib.load("rf_zone3.pkl")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    temp = float(data['Temperature'])
    hum = float(data['Humidity'])
    wind = float(data['WindSpeed'])

    features = np.array([[temp, hum, wind]])
    zone1 = model_zone1.predict(features)[0]
    zone2 = model_zone2.predict(features)[0]
    zone3 = model_zone3.predict(features)[0]
    total = zone1 + zone2 + zone3

    return jsonify({
        "Zone1": round(zone1, 2),
        "Zone2": round(zone2, 2),
        "Zone3": round(zone3, 2),
        "Total": round(total, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
