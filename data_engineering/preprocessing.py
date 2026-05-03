import pandas as pd

def clean_data(input_path="data/raw/stock_data", output_path="data/processed/clean_data.parquet"):
    """
    Clean the raw data: fill missing values, format dates.
    """
    print(f"Cleaning data from {input_path}...")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Error reading {input_path}: {e}")
        return None
    
    df.columns = [c.lower() for c in df.columns]
    
    # Forward fill then backward fill for missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    df.to_parquet(output_path, index=False)
    print(f"Saved clean data to {output_path}")
    return df

if __name__ == "__main__":
    clean_data()
