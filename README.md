# E-Commerce_Product_Delivery
**Project Overview**
This project aims to build a machine learning system that predicts whether an e-commerce product will be delivered on time or delayed.
The dataset belongs to a global electronics-based e-commerce company and contains customer, product, and shipment-related details.


**The project includes:**
Data loading & cleaning
Exploratory Data Analysis (EDA)
Feature Engineering
Model building with multiple ML algorithms
Model comparison and accuracy evaluation
Final insights and business understanding


**Major Libraries**
**Data Handling & Analysis**
import pandas as pd
import numpy as np

**Visualization**
import matplotlib.pyplot as plt
import seaborn as sns

**Preprocessing**
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

**Machine Learning Models**
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

**Evaluation Metrics**
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


**Workflow: Step-by-Step Process**
**1. Importing Libraries**
All required data analysis & ML libraries were imported before starting the pipeline.

**2. Loading the Dataset**
df = pd.read_csv('E_Commerce.csv')
Then, Checked shape, columns, null values, duplicates, and data quality.

**3. Exploratory Data Analysis (EDA)**
**Performed visual and statistical analysis:**
Distribution of shipment modes
Product weight vs delivery outcome
Countplot of Customer Ratings
Warehouse block performance
Discount vs on-time delivery
Correlation heatmap
Pairplot of major numerical features
**Insights observed:**
Heavy products had higher delivery delays.
Customers making more calls tended to experience late deliveries.
Higher prior purchases → more on-time deliveries.
Discounts above 10% were linked to better delivery performance.

**4. Data Preprocessing**
Encoding categorical variables using LabelEncoder
Scaling numerical features using StandardScaler
Splitting dataset into train-test (80–20)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

**5. Model Building**
**Trained four supervised ML models:**
Logistic Regression
KNN Classifier
Decision Tree Classifier
Random Forest Classifier
Each model trained using:
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# accuracy_score(y_test, y_pred)


**Model Accuracy Comparison**
| Model                        | Accuracy |
| ---------------------------- | -------- |
| **Decision Tree Classifier** | **69%**  |
| Random Forest Classifier     | 68%      |
| Logistic Regression          | 67%      |
| KNN Classifier               | 65%      |
Best Performing Model: Decision Tree with 69% accuracy.


**Classification Report**
Precision, Recall, F1-score generated for each model
Confusion matrix plotted for visual evaluation


**Final Conclusions**
Product weight and cost heavily influence delivery time.
Customers with more than 3 prior purchases mostly received products on time.
Higher customer care calls → more delays, indicating service issues.
Warehouse F handled the majority of shipments via ship, likely close to a seaport.
Discounts between 0–10% had higher late deliveries compared to higher discounts.
The Decision Tree model performed best, achieving 69% accuracy, making it suitable for deployment and interpretability.


**Future Enhancements**
Hyperparameter tuning (GridSearchCV)
Feature selection using Lasso/RandomForest importance
Implementing advanced models (XGBoost, LightGBM)
Deploying the best model using Flask/Streamlit
