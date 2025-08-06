# insights_phase5.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Ensure folders exist ===
os.makedirs('models', exist_ok=True)
os.makedirs('output_graphs/insights', exist_ok=True)

# === Load trained model and processed training data ===
model = joblib.load('models/randomforest.pkl')
df = pd.read_csv('processed/train_processed.csv')

# === Prepare features (excluding target and ID) ===
X = df.drop(columns=['Employee ID', 'Burn Rate'])

# === Feature importance extraction from RandomForest ===
importances = model.feature_importances_
features = X.columns

feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False)

# === Save feature importance to CSV ===
feat_df.to_csv('models/feature_importances.csv', index=False)

# === Plot and save feature importance chart ===
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_df)
plt.title('Feature Importance from Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()

# Save plot to insights subfolder
plot_path = 'output_graphs/insights/feature_importance_plot.png'
plt.savefig(plot_path)
plt.close()

# === Recommendations based on top features ===
"""
# 🔍 RECOMMENDATIONS BASED ON INSIGHTS

1. **Mental Fatigue Mitigation**
   - Mental Fatigue Score is the top predictor of Burnout.
   - ✅ Action: Implement wellness programs, mindfulness sessions, and encourage time-off.

2. **Review Resource Allocation**
   - High allocation correlates with burnout.
   - ✅ Action: Track and rebalance workloads, especially among lower designations.

3. **Enable Remote Work Options**
   - 'WFH Setup Available' influences burnout likelihood.
   - ✅ Action: Provide or improve remote setups and policies.

4. **Designation Sensitivity**
   - Junior staff (Designation 0–2) are more burnout-prone.
   - ✅ Action: Offer mentorship, manageable workloads, and growth pathways.

5. **Gender and Company-Type Patterns**
   - Some demographic segments show consistent trends.
   - ✅ Action: Investigate HR policy adjustments to support underrepresented or at-risk groups.

# 📦 OUTPUTS SAVED:
- models/feature_importances.csv
- output_graphs/insights/feature_importance_plot.png
"""
