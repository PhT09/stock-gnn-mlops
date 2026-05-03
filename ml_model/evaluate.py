import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    prob_predictions = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else predictions
    
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, prob_predictions)
    
    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "auc_roc": auc
    }
    
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
        
    return metrics
