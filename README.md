# WeatherAUS---Logistic-Regression-Analysis
A beginner friendly Logistic Regression Analysis of the WeathAUS dataset for anyone looking to familiarize with imputing, scaling, pipelining, and visual analysis. 

Weather Prediction: Beginner Logistic Regression Analysis
This project demonstrates a complete machine learning pipeline for binary classification. Using the Rain in Australia dataset, the script performs end-to-end data processing—from raw data ingestion and Exploratory Data Analysis (EDA) to building and evaluating a Logistic Regression model.

Project Highlights
Comprehensive EDA & Visual Analysis: Before modeling, the script conducts a deep dive into the data using Seaborn and Matplotlib. This includes histograms for numerical distribution, countplots for categorical comparisons, heatmaps to identify correlation, and pairplots to visualize feature relationships.

Robust Preprocessing Pipeline: The project utilizes a Scikit-Learn Pipeline to streamline data transformation. This ensures a clean workflow by handling KNN Imputing (to fill missing values based on nearest neighbors) and Standard Scaling (to normalize feature scales) in one cohesive block.

Automated Data Cleaning: * Systematic identification of NaN values.

Feature selection by dropping high-nullity columns (Evaporation, Sunshine).

Manual encoding for binary targets and One-Hot Encoding for categorical location/wind variables.

Binary Classification Model: Implementation of a Logistic Regression model using the liblinear solver, optimized for smaller datasets and binary targets.

Performance Evaluation: The model is assessed using a wide array of metrics to provide a transparent look at its predictive power:

Accuracy, Precision, Recall, and F1-Score.

Confusion Matrix for error analysis.

ROC Curve and AUC (Area Under Curve) calculation and visualization.

Coefficient Analysis to interpret which features most influence the prediction of rain.

Technical Stack
Data Handling: Pandas, NumPy

Machine Learning: Scikit-Learn

Visualization: Matplotlib, Seaborn

How to Use
Clone the repo: git clone https://github.com/yourusername/logistic-regression-weather.git

Install dependencies: pip install pandas scikit-learn matplotlib seaborn

Update File Path: Ensure the weatherAUS.csv path in the script matches your local directory.

Run: Execute python logistic_regg.py to see the visual analysis and model metrics.
