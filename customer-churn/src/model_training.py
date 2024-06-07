from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from data_preprocessing import load_data, preprocess_data, split_data

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy}')
    return accuracy

if __name__ == "__main__":
    df = load_data('data/customers.csv')
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    
    joblib.dump(model, 'models/churn_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
