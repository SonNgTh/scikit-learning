from src.data_loader import load_data
from src.train_model import train_model
from src.evaluate import evaluate_model

def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names = load_data()

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    print("\nModel Evaluation:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
