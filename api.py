import requests

url = 'http://127.0.0.1:5000/predict'
r = requests.post(url,json={"CreditScore":600,"Age":32, "Tenure":10, "Balance":645333, "NumOfProducts":6, "HasCrCard":1,"IsActiveMember":1, "EstimatedSalary":746443, "Geography_France":0, "Geography_Spain":0, "Geography_Germany":1, "Gender_Female":0, "Gender_Male":1})
print(r.json())
