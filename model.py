"""Created on Fri Jan 31 11:33:41 2020 @author: Sandeep Singh"""
# Part1 - Data Preporcessing and Importing Libraries
import pandas as pd
import numpy as np
import pickle

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Importing feature-engine - Categorical Variable
from feature_engine.categorical_encoders import OneHotCategoricalEncoder

# OneHotCategoricalEncoder?

df = pd.read_csv("churn_modelling_data.csv")
df.head()

df.describe()
df.shape
df.columns
df.dtypes
df.index

ohe =OneHotCategoricalEncoder(top_categories=None,variables=["Geography","Gender"],drop_last=False)
df1 = ohe.fit_transform(df)
df1.shape
df1.head()
df1.columns

df1.columns
df1.shape

X = df1.drop(axis=1,columns=['RowNumber', 'CustomerId', 'Surname','Exited'], errors='raise')
X.columns
X.shape

y = df1["Exited"]
y

# Splitting the dataset into Training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2,random_state = 0)

X_train.columns
X_train.shape
X_test
X_test.shape
y_train.shape
y_test.shape

# Features Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train1 = sc.fit_transform(X_train)
X_train1

X_test1 = sc.transform(X_test)
X_test1

# Part2 - Initialising & making the ANN
classifier = Sequential()

# part3 - Making the prediction and Evaluating the Model

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# =============================================================================
# model = classifier.fit(X_train1, y_train, batch_size = 10, epochs = 100)
# =============================================================================

model = classifier
model.fit(X_train1, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred

# Making the Confusion Matrix
# Classification metrics to measure the performance of the model
from  sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (16,8))
sns.heatmap(cm, annot= True, fmt="d")

cr = classification_report(y_test, y_pred)
print(cr)


# Part4- Deployemnt part1

# Predicting from the Model
# Use our ANN model to predict if the customer with the following informations will leave the bank:

X_train.shape
X_train.columns

# Credit Score: 600
# Age: 40 years old
# Tenure: 3 years
# Balance: $60000
# Number of Products: 2
# HasCrCard/Does this customer have a credit card ? Yes
# IsActiveMember/Is this customer an Active Member: Yes
# Estimated Salary: $50000
# Geography_France: France
# Geography_Spain
# Geography_Germany
# Gender_Female:
# Gender_Male: Male

# So should we say goodbye to that customer?

#X_train.columns  - SC - StandardScaler
data = sc.transform(np.array([[600,40,3,60000,2,1,1,50000,1,0,0,0,1]]))

new_prediction = model.predict(data)
new_prediction = (new_prediction > 0.5)
new_prediction
print(new_prediction, "Means Customer will not leave the Bank")


# Part5 - Deployemnt part2

# Save the model to disk
filename = 'dl_model_pickle'
pickle.dump(model, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(data)
print(result)

# =============================================================================

# Scoring on the Unseen Data

X_validate = pd.read_csv("unseen_data.csv")

X_validate

X_validate.shape
X_validate.columns

type(X_validate.values)
X_validate.dtypes

X_score = sc.transform(X_validate)
X_score


y_score = model.predict(X_score)
y_score

# Storing the predicion result in the csv file

pd.DataFrame(y_score).to_csv("y_score.csv")
