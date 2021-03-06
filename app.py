"""Created on Mon Feb  3 10:19:20 2020 @author: Sandeep Singh"""
from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

app = Flask(__name__)

deeplearning_model = open("dl_model_pickle", "rb")
ml_model = joblib.load(deeplearning_model)


@app.route("/")
def home():
    return render_template('deeplearning.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    CreditScore = float(request.form['CreditScore'])
    Age = float(request.form["Age"])
    Tenure = float(request.form["Tenure"])
    Balance = float(request.form["Balance"])
    NumOfProducts = float(request.form["NumOfProducts"])
    HasCrCard = float(request.form["HasCrCard"])
    IsActiveMember = float(request.form["IsActiveMember"])
    EstimatedSalary = float(request.form["EstimatedSalary"])
    Geography_France = float(request.form["Geography_France"])
    Geography_Spain = float(request.form["Geography_Spain"])
    Geography_Germany = float(request.form["Geography_Germany"])
    Gender_Female = float(request.form["Gender_Female"])
    Gender_Male = float(request.form["Gender_Male"])
    pred_args = [[CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_France, Geography_Spain, Geography_Germany, Gender_Female, Gender_Male]]
    #pred_args_arr = np.array(pred_args)
    #pred_args_arr = pred_args_arr.reshape(1, -1)
    #pred_args_arr = sc.fit_transform(np.array([[600,40,3,60000,2,1,1,50000,1,0,0,0,1]]))
    pred_args_arr = sc.fit_transform(np.array(pred_args))
    # load the model from disk
    #loaded_model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.predict(pred_args_arr)
    #result = round(float(result), 2)
    model_prediction = ml_model.predict(pred_args_arr)
    model_prediction = round(float(model_prediction), 2)
    percentage_format = format(model_prediction, ".2%")
    return render_template('predict.html', prediction = percentage_format)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
