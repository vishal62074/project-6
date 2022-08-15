import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle


app = Flask(__name__)
model0 = pickle.load(open('Logistic-modelproject-6.pkl','rb'))
model1 = pickle.load(open('Naive_Bayes-modelproject-6.pkl','rb'))
model2 = pickle.load(open('SVM-modelproject-6.pkl','rb'))
model3 = pickle.load(open('Random_Forest-modelproject-6.pkl','rb'))
model4 = pickle.load(open('Decision_Tree-modelproject-6.pkl','rb'))
model5 = pickle.load(open('KNN-modelproject-6.pkl','rb'))


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])

def predict():
    
    Tenth = float(request.args.get('10th'))
    Twelth = float(request.args.get('12th'))
    BTech = float(request.args.get('B.Tech'))
    SeventhSEM = float(request.args.get('7-SEM'))
    SixthSEM = float(request.args.get('6-SEM'))
    FifthSEM = float(request.args.get('5-SEM'))
    FinalPerformance = float(request.args.get('Final Performance'))
    Medium = int(request.args.get('Medium'))

# CreditScore	Geography	Gender	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary
    Model = (request.args.get('Model'))
    
    if Model=='Logistic Prediction':
      prediction = model0.predict([[Tenth,Twelth,BTech,SeventhSEM,SixthSEM,FifthSEM,FinalPerformance,Medium]])
    
    elif Model=='Naive Bayes Prediction':
      prediction = model1.predict([[Tenth,Twelth,BTech,SeventhSEM,SixthSEM,FifthSEM,FinalPerformance,Medium]])
    
    elif Model=='SVM Prediction':
      prediction = model2.predict([[Tenth,Twelth,BTech,SeventhSEM,SixthSEM,FifthSEM,FinalPerformance,Medium]])

    elif Model=='Random Forest Prediction':
      prediction = model3.predict([[Tenth,Twelth,BTech,SeventhSEM,SixthSEM,FifthSEM,FinalPerformance,Medium]])

    elif Model=='Decision Tree Prediction':
      prediction = model4.predict([[Tenth,Twelth,BTech,SeventhSEM,SixthSEM,FifthSEM,FinalPerformance,Medium]])
    
    else:
      prediction = model5.predict([[Tenth,Twelth,BTech,SeventhSEM,SixthSEM,FifthSEM,FinalPerformance,Medium]])

    
    if prediction == [1]:
      text = "Student is Placed"
    elif prediction == [0]:
        text = "Student is not Placed"
    else:
        text = "invalid"

    return render_template('index.html', prediction_text='Logistic Regression Algorithm Prediction: {}'.format(text))

if __name__=="__main__":
  app.run(debug=True)
