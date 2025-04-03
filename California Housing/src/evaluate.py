from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of the regression model.

    Parameters:
    - model: Trained regression model
    - X_test: Testing features
    - y_test: True values for testing set

    Returns:
    - metrics: Dictionary containing evaluation metrics
    """
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    return {
        'MSE': mse,
        'MAE': mae,
        'R2 Score': r2
    }
