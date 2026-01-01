from sklearn.metrics import (accuracy_score, confusion_matrix,
                             roc_curve, roc_auc_score, f1_score, precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt # hg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
df = pd.read_csv(r"D:\SNU Files\Data Science & Machine Learning\Assignments\weatherAUS.csv")
print("Data specifications: ",df.head(), df.info(),df.describe(),  sep="\n")
print("\nNo. of NaN vlaue columns: ", df.isna().sum(), "\nIn total: ", df.isna().sum().sum())

# Replacing Yes/No with 1/0
b = {'Yes': 1, 'No': 0}
df[['RainToday', 'RainTomorrow']] = (df[['RainToday', 'RainTomorrow']].replace(b)).fillna(0).astype(int) #making in place changes
#Checking if nulls exist
if df.isna().sum().sum() == 0:
    print("\nNo NaN values found => Imputing not needed.")
# if nulls do exist, going aead with Preprocessing & Regression
else:
    print("\nNullls found (Numeric)=> Imputing => Scaling")
    df1 =df.drop(["Date", "Evaporation", "Sunshine"], axis=1) #Remove columns entirely comprising of nulls + the date column
    df_num = df1.select_dtypes(exclude=['object']); l=list(df_num.columns)
    #histograms for numeric features alone
    
    num_features = df_num.shape[1] #1st index of a tupple containing (no.rows, no.columns) of df_num
    fig, ax = plt.subplots(nrows=int(num_features/3) + 1, ncols=3, # Calculating rows required based on features (3 columns per row)
                             figsize=(15, 5 * (int(num_features/3) + 1)))
    ax = ax.flatten() # Flattens the 2D array of ax to 1D for easy iteration
    for i, col in enumerate(df_num.columns):
        sns.histplot(data=df_num, x=col, bins=50, kde=True, ax=ax[i])
        ax[i].set_title(f'Distribution of {col}', fontsize=14)
        ax[i].set_xlabel(col)
    #countplot for categorical features. using aubplot axes 16 & 17 as they are empty
    sns.countplot(data=df1, x="Location", hue="RainToday", ax=ax[16])
    ax[16].set_title("Comparison Countplot")
    sns.countplot(data=df1, x="Location", hue="RainTomorrow", ax=ax[17])
    ax[17].set_title("Comparison Countplot")
    plt.tight_layout()
    plt.show()
    
    #heatmap for numerical dataframe
    plt.figure(figsize=(10, 10))
    sns.heatmap(data=df_num)
    plt.title("Comparison heatmap")
    plt.show()
    
    #pairplots for initially modified dataframe
    plt.figure(figsize=(10, 10))
    sns.pairplot(df1,hue="Location", diag_kind="kde" )
    plt.title("Pair plot")
    plt.show()
    
    #Creating a pipeline to easily preprocess and scale the numeric dataframe
    num_pl = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5)), #Taking approx neighbours as sqrt(No. of Rows))
                                       ('scaler', StandardScaler())])
    num_pl.set_output(transform="pandas")
    df_num_processed = num_pl.fit_transform(df_num)
    print("\n--- Processed Numeric Data Info ---\n")
    df_num_processed.info()
    df_num_processed = df_num_processed.reset_index(drop=True)
    #One-hot encoding the object datatypes, and dropping the first column for each through dummy variables
    df_obj = df1.select_dtypes(exclude=['float64', 'int64'])
    l=list(df_obj.columns)
    df_obj_processed = pd.get_dummies(df_obj, columns=l, drop_first=True)
    print("\n--- Processed Categorical Data (Dummy Variables) Info ---\n")
    df_obj_processed.info()
    df_obj_processed = df_obj_processed.reset_index(drop=True)
    dfin = pd.concat([df_num_processed, df_obj_processed], axis=1)
    
    #Combining cleaned + preprocessed dataframe
    print("\n--- Final Combined DataFrame Info ---\n")
    dfin.info() 
    
    # Logistic Regression
    X = dfin.drop("RainTomorrow", axis=1)
    y = dfin["RainTomorrow"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("Using Iris dataset (binary) for demonstration.")    
    # Train Logistic Regression Model
    # C=1.0 is the default. max_iter=1000 for convergence.
    model = LogisticRegression(solver='liblinear', C=1.0, random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    # Predict on the Test Set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probability of the positive class
    
    # Compute Metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print("\n--- Model Evaluation Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_mat)
    
    # Plot ROC Curve and Compute AU
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='orange', label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Display and Interpret Coefficients
    coeff = pd.Series(model.coef_[0], index=list(dfin.columns)).sort_values(ascending=False)
    print("\nCoefficients:")
    print(coeff)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
