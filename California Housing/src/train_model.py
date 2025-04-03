from sklearn.linear_model import LinearRegression
import joblib
import os

def train_model(X_train, y_train, model_dir='models'):
    """
    Train a Linear Regression model and save it to disk.

    Parameters:
    - X_train: Training features
    - y_train: Training target
    - model_dir: Directory to save the trained model

    Returns:
    - model: Trained Linear Regression model
    """
    # Initialize the model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Save the trained model
    model_path = os.path.join(model_dir, 'linear_regression_model.pkl')
    joblib.dump(model, model_path)

    return model
