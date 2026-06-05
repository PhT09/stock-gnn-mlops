from data_engineering.preprocessing import preprocess

def engineer_features():
    """Engineer features from raw stock data."""
    print("🔄 Running feature engineering...")
    preprocess()
    print("✅ Feature engineering completed!")

if __name__ == "__main__":
    engineer_features()
