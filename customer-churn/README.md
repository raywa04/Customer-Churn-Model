# Customer Churn Prediction

## Project Description

This project aims to develop a predictive model to identify customers likely to churn from a subscription service or product.

## Features

1. Data collection from customer usage logs.
2. Feature engineering to identify relevant factors.
3. Training and evaluating models like logistic regression.

## Technology Stack

- Python
- Flask
- Scikit-learn
- Pandas
- Joblib

## Setup Instructions

1. Install Python and pip.
2. Navigate to the `customer-churn` folder.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run the model training script: `python src/model_training.py`.
5. Start the Flask app: `python src/app.py`.

## Usage

Send a POST request to `/predict` with JSON data containing the features of a customer.

Example:
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"data": [1, 0, 0, 45, 1, 0, 70, 0]}'
```
