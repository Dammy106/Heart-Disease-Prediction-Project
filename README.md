# Heart-Disease-Prediction-Project
Using two Machine Learning Models to predict heart disease.
# Overview
This project focuses on predicting heart disease using the Heart Disease UCI dataset from Kaggle. I used two machine learning models: Random Forest Classifier and Logistic Regression to classify whether a person has heart disease based on various health indicators. Both the jupyter notebook and heart dataset are provided.<br/>

# Steps Followed

## Project Steps
### Data Loading:

1.) Loaded and explored the dataset to understand the distribution of features and target values.
### Data Cleaning & Preprocessing:

#### Performed necessary preprocessing steps
1.) Analyzed data to check if dataset had missing values.<br/>
2.) Converted categorical variables using one-hot encoding for columns like 'cp' (chest pain type) and 'thal.'
### Train-Test Split:

1.) Split the dataset into 80% training and 20% testing to ensure proper evaluation.
### Feature Scaling

1.) Standardized numerical features using StandardScaler to bring them to the same scale.
### Model Selection

### Chose two models for comparison
1.) Random Forest Classifier<br/>
1.) Logistic Regression
### Training the Models

1.) Trained both models using the training set.
### Model Evaluation

Evaluated the performance on the validation set (from a split of the training set).<br/>
Random Forest Validation Accuracy: 99.39%<br/>
Logistic Regression Validation Accuracy: 90.85%<br/>
### Overfitting Checks

Used techniques like cross-validation and learning curves to check for overfitting in the Random Forest model.
## Model Performance

Evaluated both models using accuracy, confusion matrices, and classification reports.<br/>
### Random Forest Results:
Validation Accuracy: 99.39%<br/>
Confusion Matrix:
[[79, 1], [0, 84]]<br/>
Classification Report:
Precision, Recall, F1-score: ~99% across both classes.<br/>
### Logistic Regression Results:
Validation Accuracy: 90.85%<br/>
Confusion Matrix:
[[67, 13], [2, 82]]<br/>
Classification Report:
Precision, Recall, F1-score for Class 1: ~92%.<br/>
## Next Steps:
1.) Possible improvement through hyperparameter tuning for Random Forest.<br/>
2.) Explore other models or ensemble methods for better predictions.<br/>

### Visualizations
Visualizations of features, correlations, and feature importance can be done using libraries like Matplotlib and Seaborn.<br/>
Code for the correlation heatmap: import seaborn as sns
import matplotlib.pyplot as plt

### Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
