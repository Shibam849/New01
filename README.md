Titanic Survival Prediction

Task Overview

This project develops a machine learning model to predict whether a passenger survived the Titanic disaster using features like age, gender, ticket class, fare, etc. The dataset is preprocessed, a Random Forest Classifier is trained, and performance is evaluated using accuracy, precision, recall, and F1-score.

Dataset

The dataset (tested.csv) contains the following columns:





PassengerId: Unique identifier (dropped)



Survived: Target variable (0 = Did not survive, 1 = Survived)



Pclass: Ticket class (1, 2, or 3)



Name: Passenger name (dropped)



Sex: Gender (male/female)



Age: Age in years



SibSp: Number of siblings/spouses aboard



Parch: Number of parents/children aboard



Ticket: Ticket number (dropped)



Fare: Ticket fare



Cabin: Cabin number (dropped due to many missing values)



Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

Approach





Preprocessing:





Dropped irrelevant columns: PassengerId, Name, Ticket, Cabin.



Handled missing values: Imputed Age and Fare with median, Embarked with mode.



Encoded categorical variables: Sex (male=0, female=1), Embarked (one-hot encoding).



Normalized numerical features (Age, Fare) using StandardScaler.



Model Selection: Used a Random Forest Classifier due to its ability to handle non-linear relationships and robustness to overfitting.



Training: Split the data into 80% training and 20% testing sets. Trained the model on the training set.



Evaluation: Evaluated the model on the test set using accuracy, precision, recall, and F1-score.

Code Structure





titanic_survival_prediction.py: Main script for preprocessing, model training, and evaluation.

Results





Accuracy: Measures overall correctness.



Precision: Proportion of predicted survivors who actually survived.



Recall: Proportion of actual survivors correctly predicted.



F1-Score: Harmonic mean of precision and recall.

Run the script to see the exact metrics. The Random Forest model typically achieves strong performance due to its ensemble nature.

How to Run





Ensure Python and required libraries (pandas, numpy, scikit-learn) are installed.



Place tested.csv in the same directory as the script.



Run the script:

python titanic_survival_prediction.py

Future Improvements





Experiment with other models like Logistic Regression or XGBoost.



Perform feature engineering, e.g., create a "FamilySize" feature from SibSp and Parch.



Use cross-validation for more robust evaluation.
