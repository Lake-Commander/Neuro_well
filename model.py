import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Create models directory if not exists
os.makedirs('models', exist_ok=True)

# Load processed training data
train_df = pd.read_csv('processed/train_processed.csv')

# Drop rows where target is missing
train_df = train_df.dropna(subset=['Burn Rate'])

# Separate features and target
X = train_df.drop(columns=['Burn Rate', 'Employee ID'])
y = train_df['Burn Rate']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_model_name = ""
best_mse = float('inf')

# Train, evaluate, and save models
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = model.predict(X_val)

    # Evaluate
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"{name} MSE: {mse:.4f}")
    print(f"{name} R²: {r2:.4f}")

    # Save model
    model_path = f'models/{name.lower()}.pkl'
    joblib.dump(model, model_path)
    print(f"{name} saved to {model_path}")

    # Track best model
    if mse < best_mse:
        best_mse = mse
        best_model = model
        best_model_name = name

# Save the best model separately
if best_model:
    best_model_path = 'models/best_model.pkl'
    joblib.dump(best_model, best_model_path)
    print(f"\n✅ Best model is {best_model_name} with MSE: {best_mse:.4f}")
    print(f"📦 Saved as {best_model_path}")
