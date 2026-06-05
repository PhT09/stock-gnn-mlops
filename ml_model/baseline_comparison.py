"""
Baseline Models Comparison Script

Compares XGBoost with baseline models:
- Logistic Regression
- Random Forest
- LSTM

Generates comparison table and visualizations.
"""

import pandas as pd
import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# For LSTM
try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    print("⚠️  TensorFlow not available, LSTM will be skipped")
    HAS_TENSORFLOW = False

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Databricks


def load_data(data_path):
    """Load and prepare data exactly like XGBoost training"""
    # Fallback to local downloaded_data if Databricks volume path does not exist
    if not os.path.exists(data_path):
        local_path = "downloaded_data"
        if os.path.exists(local_path):
            print(f"⚠️ Databricks volume path '{data_path}' not found. Falling back to local: '{local_path}'")
            data_path = local_path
            
    print(f"\n📂 Loading data from: {data_path}")
    
    try:
        df = pd.read_parquet(data_path)
        print(f"   ✅ Loaded {len(df):,} rows")
        
        # Drop rows with null targets (latest day's data)
        df_train = df.dropna(subset=['target']).copy()
        
        # Extract features from Spark ML DenseVector
        print("   🔄 Extracting features from DenseVector...")
        X = np.vstack(df_train['scaled_features'].apply(lambda x: x['values']).values)
        y = df_train['target'].values
        
        # Time-based split (80/20, no shuffle)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"   ✅ Train: {len(X_train):,} samples")
        print(f"   ✅ Test:  {len(X_test):,} samples")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"   ❌ Error loading data: {str(e)}")
        raise


def evaluate_model(y_true, y_pred, y_prob):
    """Calculate metrics (same as XGBoost)"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob))
    }


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train Logistic Regression baseline"""
    print("\n1️⃣  Training Logistic Regression...")
    
    start_time = time.time()
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    training_time = time.time() - start_time
    
    print(f"   ✅ Completed in {training_time:.2f}s")
    print(f"   📊 Accuracy: {metrics['accuracy']:.4f}")
    print(f"   📊 F1-Score: {metrics['f1_score']:.4f}")
    print(f"   📊 AUC-ROC:  {metrics['auc_roc']:.4f}")
    
    return metrics, training_time


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest baseline"""
    print("\n2️⃣  Training Random Forest...")
    
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    training_time = time.time() - start_time
    
    print(f"   ✅ Completed in {training_time:.2f}s")
    print(f"   📊 Accuracy: {metrics['accuracy']:.4f}")
    print(f"   📊 F1-Score: {metrics['f1_score']:.4f}")
    print(f"   📊 AUC-ROC:  {metrics['auc_roc']:.4f}")
    
    return metrics, training_time


def train_lstm(X_train, X_test, y_train, y_test):
    """Train simple LSTM baseline"""
    if not HAS_TENSORFLOW:
        print("\n3️⃣  LSTM: Skipped (TensorFlow not available)")
        return None, 0
    
    print("\n3️⃣  Training LSTM...")
    
    start_time = time.time()
    
    # Reshape for LSTM: (samples, timesteps, features)
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    # Simple LSTM model
    model = Sequential([
        LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping to save time
    early_stop = EarlyStopping(
        monitor='loss',
        patience=3,
        restore_best_weights=True,
        verbose=0
    )
    
    # Train (max 10 epochs for speed)
    print("   🔄 Training (max 10 epochs)...")
    model.fit(
        X_train_lstm, y_train,
        epochs=10,
        batch_size=64,
        verbose=0,
        callbacks=[early_stop]
    )
    
    # Predict
    y_prob = model.predict(X_test_lstm, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    
    metrics = evaluate_model(y_test, y_pred, y_prob)
    training_time = time.time() - start_time
    
    print(f"   ✅ Completed in {training_time:.2f}s")
    print(f"   📊 Accuracy: {metrics['accuracy']:.4f}")
    print(f"   📊 F1-Score: {metrics['f1_score']:.4f}")
    print(f"   📊 AUC-ROC:  {metrics['auc_roc']:.4f}")
    
    return metrics, training_time


def load_xgboost_metrics(metrics_path):
    """Load existing XGBoost metrics"""
    print("\n4️⃣  Loading XGBoost metrics...")
    
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        
        metrics = data['metrics']
        print(f"   ✅ Loaded existing metrics")
        print(f"   📊 Accuracy: {metrics['accuracy']:.4f}")
        print(f"   📊 F1-Score: {metrics['f1_score']:.4f}")
        print(f"   📊 AUC-ROC:  {metrics['auc_roc']:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"   ⚠️  Could not load XGBoost metrics: {str(e)}")
        print(f"   Using placeholder values")
        return {
            "accuracy": 0.0,
            "f1_score": 0.0,
            "auc_roc": 0.0
        }


def create_comparison_table(results):
    """Create and print formatted comparison table"""
    print("\n" + "="*90)
    print("📊 BASELINE MODELS COMPARISON RESULTS")
    print("="*90)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df.round(4)
    
    # Print formatted table
    print("\n" + df.to_string(index=False))
    
    # Highlight best model
    print("\n" + "="*90)
    best_model = df.loc[df['AUC-ROC'].idxmax(), 'Model']
    best_auc = df['AUC-ROC'].max()
    print(f"🏆 BEST MODEL: {best_model} (AUC-ROC: {best_auc:.4f})")
    print("="*90)
    
    return df


def create_visualizations(df, output_path):
    """Create comparison bar charts"""
    print(f"\n📈 Creating visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['Accuracy', 'F1-Score', 'AUC-ROC']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        bars = ax.bar(df['Model'], df[metric], color=colors)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved to: {output_path}")
    
    plt.close()


def save_results(results, output_path):
    """Save results to JSON"""
    print(f"\n💾 Saving results...")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✅ Saved to: {output_path}")


def main():
    """Main execution"""
    print("="*90)
    print("🚀 STARTING BASELINE MODELS COMPARISON")
    print("="*90)
    
    # Paths - READ DIRECTLY FROM VOLUME (where real data is)
    data_path = "/Volumes/workspace/default/stock_data/processed/stock_features.parquet"
    workspace_root = "/Workspace/Users/vphat545@gmail.com/stock-gnn-mlops"
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        xgboost_metrics_path = f"{workspace_root}/models/best_metrics.json"
        results_json = f"{workspace_root}/models/baseline_comparison_results.json"
        results_png = f"{workspace_root}/models/baseline_comparison.png"
    else:
        xgboost_metrics_path = "models/best_metrics.json"
        results_json = "models/baseline_comparison_results.json"
        results_png = "models/baseline_comparison.png"
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_data(data_path)
        
        # Train baseline models
        lr_metrics, lr_time = train_logistic_regression(X_train, X_test, y_train, y_test)
        rf_metrics, rf_time = train_random_forest(X_train, X_test, y_train, y_test)
        lstm_metrics, lstm_time = train_lstm(X_train, X_test, y_train, y_test)
        
        # Load XGBoost metrics
        xgb_metrics = load_xgboost_metrics(xgboost_metrics_path)
        
        # Compile results
        results = [
            {
                "Model": "Logistic Regression",
                "Accuracy": lr_metrics['accuracy'],
                "F1-Score": lr_metrics['f1_score'],
                "AUC-ROC": lr_metrics['auc_roc'],
                "Training Time (s)": round(lr_time, 2)
            },
            {
                "Model": "Random Forest",
                "Accuracy": rf_metrics['accuracy'],
                "F1-Score": rf_metrics['f1_score'],
                "AUC-ROC": rf_metrics['auc_roc'],
                "Training Time (s)": round(rf_time, 2)
            }
        ]
        
        # Add LSTM if available
        if lstm_metrics is not None:
            results.append({
                "Model": "LSTM",
                "Accuracy": lstm_metrics['accuracy'],
                "F1-Score": lstm_metrics['f1_score'],
                "AUC-ROC": lstm_metrics['auc_roc'],
                "Training Time (s)": round(lstm_time, 2)
            })
        
        # Add XGBoost
        results.append({
            "Model": "XGBoost ⭐",
            "Accuracy": xgb_metrics['accuracy'],
            "F1-Score": xgb_metrics['f1_score'],
            "AUC-ROC": xgb_metrics['auc_roc'],
            "Training Time (s)": "N/A (existing)"
        })
        
        # Create comparison table
        df = create_comparison_table(results)
        
        # Create visualizations
        create_visualizations(df, results_png)
        
        # Save results
        save_results(results, results_json)
        
        print("\n" + "="*90)
        print("✅ COMPARISON COMPLETED SUCCESSFULLY")
        print("="*90)
        print(f"\n📁 Output files:")
        print(f"   • Table & metrics: {results_json}")
        print(f"   • Visualizations:  {results_png}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
