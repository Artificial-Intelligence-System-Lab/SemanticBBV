import os
import numpy as np
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

RESULTS_DIR = "./inference_results"

def extract_epoch(filename):
    match = re.search(r'epoch[_\-]?(\d+)', filename)
    return int(match.group(1)) if match else -1

def evaluate_npy(pred_path, label_path):
    preds = np.load(pred_path)
    labels = np.load(label_path)
    
    if len(preds) != len(labels):
        raise ValueError(f"Prediction and label quantity mismatch: {pred_path}")
    
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    return acc, f1_macro, precision, recall

def collect_results(results_dir):
    npy_files = os.listdir(results_dir)
    label_files = sorted([f for f in npy_files if f.startswith("labels") and f.endswith(".npy")])
    
    results = []
    
    for label_file in label_files:
        epoch = extract_epoch(label_file)
        pred_file = label_file.replace("labels", "predictions")
        
        label_path = os.path.join(results_dir, label_file)
        pred_path = os.path.join(results_dir, pred_file)
        
        if not os.path.exists(pred_path):
            print(f"⚠️ Missing prediction file: {pred_path}")
            continue
        
        try:
            acc, f1, prec, rec = evaluate_npy(pred_path, label_path)
            results.append({
                "epoch": epoch,
                "accuracy": acc,
                "f1_macro": f1,
                "precision": prec,
                "recall": rec,
                "label_file": label_file,
                "pred_file": pred_file
            })
        except Exception as e:
            print(f"❌ Evaluation failed: {label_file} - {e}")
    
    return sorted(results, key=lambda x: x["f1_macro"], reverse=True)

def print_summary(results):
    print("\n📊 Model ranking results (sorted by Macro-F1 descending):")
    for i, r in enumerate(results):
        print(f"{i+1:2d}. Epoch {r['epoch']:>2}: "
              f"Acc={r['accuracy']:.4f}, "
              f"F1={r['f1_macro']:.4f}, "
              f"P={r['precision']:.4f}, "
              f"R={r['recall']:.4f} "
              f"({r['label_file']})")

if __name__ == "__main__":
    results = collect_results(RESULTS_DIR)
    print_summary(results)