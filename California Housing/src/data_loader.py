import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(test_size=0.2, random_state=42):
    """
    Load and preprocess California housing data.
    Uses local cache if available. Saves full dataset and split data to /data/.

    Returns:
    - x_train, x_test, y_train, y_test, feature_names
    """
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    dataset_path = os.path.join(data_dir, "dataset.csv")
    x_train_path = os.path.join(data_dir, "x_train.csv")
    x_test_path = os.path.join(data_dir, "x_test.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    y_test_path = os.path.join(data_dir, "y_test.csv")

    # Load or download dataset
    if os.path.exists(dataset_path):
        print(f"📂 Loading dataset from local: {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        print("🌐 Downloading dataset from sklearn...")
        df = fetch_california_housing(as_frame=True).frame
        df.to_csv(dataset_path, index=False)
        print(f"✅ Dataset saved to: {dataset_path}")

    x = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"]

    # Load cached split if exists
    if all(os.path.exists(p) for p in [x_train_path, x_test_path, y_train_path, y_test_path]):
        print("📂 Loading cached train/test split...")
        x_train = pd.read_csv(x_train_path)
        x_test = pd.read_csv(x_test_path)
        y_train = pd.read_csv(y_train_path).squeeze()
        y_test = pd.read_csv(y_test_path).squeeze()
    else:
        print("🧪 Creating new train/test split...")
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )

        # Save splits
        x_train.to_csv(x_train_path, index=False)
        x_test.to_csv(x_test_path, index=False)
        y_train.to_csv(y_train_path, index=False)
        y_test.to_csv(y_test_path, index=False)
        print("✅ Train/test split saved to disk")

    # Feature scaling
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x.columns)

    return x_train_scaled, x_test_scaled, y_train, y_test, x.columns.tolist()
