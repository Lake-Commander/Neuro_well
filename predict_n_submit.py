import pandas as pd
import joblib
import os

# === Load processed test data ===
test_df = pd.read_csv('processed/test_processed.csv')

# Keep track of Employee IDs for submission
employee_ids = test_df['Employee ID'].copy()

# Drop ID column before prediction
X_test = test_df.drop(columns=['Employee ID'])

# === Load the best model ===
model = joblib.load('models/best_model.pkl')

# === Make predictions ===
preds = model.predict(X_test)

# Clip predictions between 0 and 1
preds = preds.clip(0, 1)

# === Create submission DataFrame ===
submission = pd.DataFrame({
    'Employee ID': employee_ids,
    'Burn Rate': preds
})

# === Save to CSV ===
os.makedirs('submissions', exist_ok=True)
submission_path = 'submissions/predicted_burn_rate.csv'
submission.to_csv(submission_path, index=False)

print(f"✅ Submission file saved to '{submission_path}'")
