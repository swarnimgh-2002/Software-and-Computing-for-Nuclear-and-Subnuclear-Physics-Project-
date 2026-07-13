import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# next I import ML libraries and tools from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report

def load_data(filepath):
    """
    Loads the MAGIC Gamma Telescope dataset.
    The dataset has 10 numerical features describing the shower image ellipse.
    The 11th column is the class: 'g' for gamma (signal), 'h' for hadron (background).
    """
    print("Loading data into Pandas DataFrame...")
    
    # The dataset doesn't have a header row, so I defined the physics features manually
    columns = [
        "fLength", "fWidth", "fSize", "fConc", "fConc1", 
        "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"
    ]
    
    try:
        df = pd.read_csv(filepath, names=columns)
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        sys.exit(1)
        
    # To convert string classes to binary integers: Gamma (g) = 1 (Signal), Hadron (h) = 0 (Background)
    df['label'] = (df['class'] == 'g').astype(int)
    
    # Separate features (X) from target label (y)
    X = df.drop(['class', 'label'], axis=1)
    y = df['label']
    
    return X, y

def train_and_evaluate_model(X, y):
    """
    Trains a Random Forest model and calculates evaluation metrics.
    """
    print("Splitting data into 70% Training set and 30% Testing set...")
    # so, the ML model don't evalute on the trained data!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    print("Training Random Forest Classifier...")
    # Initialize the AI model with 100 decision trees
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    print("Evaluating model performance on unseen test data...")
    # Predict the probability that a shower is a Gamma ray
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Also get hard predictions (0 or 1) for accuracy metrics
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Hadron (Bkg)", "Gamma (Sig)"]))
    
    # Calculate data for the ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC (Area Under Curve): {roc_auc:.4f}")
    
    return rf_model, fpr, tpr, roc_auc

def generate_ml_plots(rf_model, feature_names, fpr, tpr, roc_auc, output_path):
    """
    Creates a professional 2-panel plot showing the ROC Curve and Feature Importances.
    """
    print(f"Generating ML diagnostic plots to {output_path}...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- 1: ROC Curve ---
    # The ROC curve shows the trade off between True Positive Rate and False Positive Rate
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'Random Forest (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (Background incorrectly kept)', fontsize=12)
    ax1.set_ylabel('True Positive Rate (Gamma signal correctly kept)', fontsize=12)
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=14)
    ax1.legend(loc="lower right", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # --- 2: Feature Importance ---
    # Random Forests allow us to see which physical variables were most important for classification
    importances = rf_model.feature_importances_
    # Sorting them in descending order
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    ax2.bar(range(len(importances)), sorted_importances, color='teal', align='center')
    ax2.set_xticks(range(len(importances)))
    ax2.set_xticklabels(sorted_features, rotation=45, ha='right')
    ax2.set_title('Physics Feature Importance', fontsize=14)
    ax2.set_ylabel('Relative Importance', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()
    print("Plotting complete!")

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(project_root, 'data', 'magic04.data')
    output_file = os.path.join(project_root, 'output', 'ml_evaluation_plots.pdf')
    
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # 1. Load Data
    X, y = load_data(data_file)
    feature_names = X.columns.tolist()
    
    # 2. Train AI and get metrics
    rf_model, fpr, tpr, roc_auc = train_and_evaluate_model(X, y)
    
    # 3. Generate ML Physics plots
    generate_ml_plots(rf_model, feature_names, fpr, tpr, roc_auc, output_file)

if __name__ == "__main__":
    main()
