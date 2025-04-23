import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("new_data.csv")

# Convert to binary
df['rot'] = df['rot'].map({'Yes': 1, 'No': 0})

# Drop unnecessary columns
df = df.drop(columns=['lat', 'long', 'date'])

# Split features and target
X = df.drop(columns=['rot'])
y = df['rot']

# Define feature types
numerical_cols = ['mean_temp_3m', 'mean_temp_1y', 'mean_temp_5y', 'min_temp', 'max_temp', 'humidity', 'soil_humidity']
categorical_cols = ['d_type', 'species', 'municipality']

# Preprocessing numerical
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing categorical
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine into column transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, numerical_cols),
    ('cat', cat_transformer, categorical_cols)
])

# Build XGBoost classifier
xgb_clf = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),  # weight positive class more
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

clf_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', xgb_clf)
])

# Test setup
rot_yes = df[df['rot'] == 1]
rot_no = df[df['rot'] == 0]

rot_yes_test = rot_yes.sample(n=30, random_state=42)
rot_no_test = rot_no.sample(n=270, random_state=42)

rot_yes_train = rot_yes.drop(rot_yes_test.index)
rot_no_train = rot_no.drop(rot_no_test.index)

train_df = pd.concat([rot_yes_train, rot_no_train]).sample(frac=1, random_state=42)
test_df = pd.concat([rot_yes_test, rot_no_test]).sample(frac=1, random_state=42)

# Final training and test splits
X_train = train_df.drop(columns=['rot'])
y_train = train_df['rot']

X_test = test_df.drop(columns=['rot'])
y_test = test_df['rot']

# Train model
clf_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = clf_pipeline.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


