##  Disclaimer
> This application is strictly for educational and research purposes. It is not intended for clinical use.

#  Employee Burnout Prediction

This project predicts employee burnout levels based on factors such as job designation, company type, resource allocation, mental fatigue and work-from-home setup. It uses machine learning models trained on a real-world dataset and presents predictions through a user-friendly Streamlit app.

---

##  Features

-  Exploratory Data Analysis (EDA) on burnout drivers
-  Correlation & Bivariate Visualizations
-  Machine learning models: Random Forest, Linear, Ridge and Lasso Regression
-  Performance metrics: MSE, R² Score
-  Streamlit App for real-time burnout prediction

---

##  Technologies

- Python 3.13
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

##  Run Locally

### 1. Clone the repository:

```bash
git clone https://github.com/Lake-Comander/Neuro_well.git
cd Neuro_well
```

### 2. Create a virtual environment and activate it:
Make sure you have Python 3.13 installed.
```bash
python -m venv venv
venv\\Scripts\\activate
```

### 3. Install required packages:
```bash
pip instal -r requirements.txt
```

##  Project Structure

```
burnout-prediction/
├── app.py # Streamlit web app
├── models/
│ ├── random_forest_model.pkl
│ ├── scaler.pkl
│ └── feature_order.pkl
├── output_graphs/
│ ├── correlation/ # Heatmaps and correlation matrices
│ ├── bivariate/ # Boxplots, scatterplots, etc.
├── processed_rf_selected_data.csv # Feature-selected dataset used for training
├── burnout_notebook.ipynb # Jupyter notebook (EDA + modeling)
├── requirements.txt # Required Python packages
└── README.md # Project documentation
```


##  Train Your Own Model
**To train a new model with your dataset:**

1. Prepare your dataset in CSV format. Use the sample datasets here for schema reference.

2. Open and run a3.ipynb in Jupyter Notebook.

3. Follow the steps to preprocess, train and save the model.

4. Ensure your app loads the new model path.

## Deployed App
Access the live app:
 [Click here to open the app](https://neurowell-bo.streamlit.app/).

##  Acknowledgments
This project was built under the guidance and mentorship of the 3MTT (Three Million Technical Talent) program by the National Information Technology Development Agency (NITDA), Nigeria.

We sincerely appreciate NITDA and the Federal Ministry of Communications, Innovation and Digital Economy for the opportunity to learn, grow, and contribute to Nigeria’s digital transformation journey.

Thank you for empowering Nigerian youths with the skills to build real-world solutions.



