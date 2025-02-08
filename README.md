# Bunkruptcy-Prevention
This project focuses on predicting bankruptcy using machine learning models. Below is a concise summary of the key steps involved:

# Data Loading & Preprocessing:
The dataset is loaded from an Excel file.
Basic exploratory steps like df.describe(), df.info(), and df.isnull().sum() are performed.
The target variable (class) is encoded using Label Encoding.

# Exploratory Data Analysis (EDA):
Box plots, histograms, and KDE plots are used for visualizing data distribution.
Count plots and cross-tabulation are applied to analyze categorical variables.
A correlation matrix and heatmap are used to identify relationships between features.

# Model Building:
Various machine learning models are used, including:
Logistic Regression
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Gaussian Naïve Bayes (GNB)
The dataset is split into training and testing sets using train_test_split().

# Model Evaluation:
Models are evaluated using performance metrics like accuracy, precision, recall, and confusion matrix.

# Insights & Conclusion:
The best-performing model is identified based on evaluation metrics.
The impact of different risk factors (e.g., industrial risk, management risk) on bankruptcy prediction is analyzed.
