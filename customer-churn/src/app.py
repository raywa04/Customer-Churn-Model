from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    scaled_data = scaler.transform([data])
    prediction = model.predict(scaled_data)
    return jsonify({'churn': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
