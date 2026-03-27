import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_feature_lists(df):
    """Identifies numerical and categorical columns, dropping constants."""
    # Drop target and non-informative unique columns
    cols_to_drop = ['Attrition', 'EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    
    # Only drop columns if they actually exist in the dataframe
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    return numeric_features, categorical_features

def get_preprocessing_pipeline(numeric_features, categorical_features):
    """Creates a transformer for scaling numbers and encoding text."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def load_and_split_data(filepath):
    """Loads CSV and splits into Train/Test sets."""
    df = pd.read_csv(filepath)
    
    # Convert Target to binary
    df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    
    # Stratify ensures the 16% attrition rate is preserved in both sets
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)