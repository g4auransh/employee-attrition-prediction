import pandas as pd
from src.preprocess import get_feature_lists

# Load the data
df = pd.read_csv('data/HR_Analytics.csv')

# Get the lists
num_features, cat_features = get_feature_lists(df)

print(f"✅ Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"📊 Numerical Features ({len(num_features)}): {num_features[:5]}...")
print(f"🔤 Categorical Features ({len(cat_features)}): {cat_features[:5]}...")
print(f"⚖️ Target Balance:\n{df['Attrition'].value_counts(normalize=True) * 100}")