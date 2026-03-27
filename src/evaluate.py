import sys
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Add project root to path so it can find 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def evaluate_model():
    # 1. Load the model and the test data we saved during training
    try:
        model = joblib.load('models/best_model.pkl')
        X_test = pd.read_csv('data/X_test.csv')
        y_test = pd.read_csv('data/y_test.csv')
        print("✅ Model and Test Data loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: Missing model or test data. Run src/train.py first!")
        return

    # 2. Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability of 'Leaving'

    # 3. Print Metrics
    print("\n" + "="*30)
    print("📊 CLASSIFICATION REPORT")
    print("="*30)
    # Target names 0=Stay, 1=Leave
    print(classification_report(y_test, y_pred, target_names=['Stay', 'Leave']))
    
    auc = roc_auc_score(y_test, y_proba)
    print(f"📈 ROC-AUC Score: {auc:.2f}")
    print("="*30)

    # 4. Generate Visuals (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Stay', 'Leave'], 
                yticklabels=['Stay', 'Leave'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Employee Attrition: Confusion Matrix')
    
    # Save the plot to your models folder
    plt.savefig('models/confusion_matrix.png')
    print("\n🖼️ Visual saved: 'models/confusion_matrix.png'")
    print("💡 Tip: Look at the 'Leave' Recall to see how many quitters we caught!")

    # 5. Professional Feature Importance Mapping
    # Access the preprocessor and the classifier from the pipeline
    preprocessor = model.named_steps['preprocessor']
    classifier = model.named_steps['classifier']

    # Get the feature names after one-hot encoding
    cat_features = preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out()
    num_features = preprocessor.transformers_[0][2]
    all_feature_names = list(num_features) + list(cat_features)

    # Create a DataFrame for visualization
    importances = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': classifier.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\n🚀 TOP 5 PREDICTORS OF ATTRITION:")
    print("="*30)
    print(importances.head(5).to_string(index=False))

if __name__ == "__main__":
    evaluate_model()