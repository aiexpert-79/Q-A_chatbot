from flask import Flask, render_template, request
import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import csv
import warnings
import numpy as np
from mako.template import Template

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)

training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()

# mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)

model = SVC()
model.fit(x_train, y_train)

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

serverityDictionary = dict()
description_list = dict()
precautionDictionary = dict()
symptoms_dict = {}

for index, symptoms in enumerate(x):
  symptoms_dict[symptoms] = index
  
def calc_condition(exp, days):
  sum = 0
  for item in exp:
    sum = sum + serverityDictionary[item]
  if ((sum * days) / (len(exp) + 1) > 13):
    return "You should take the consultation from doctor."
  else:
    return "It might not be that bad but you should take precautions."
  
def getDescription():
  global description_list
  with open('MasterData/symptom_Description.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
      _description = {row[0]: row[1]}
      description_list.update(_description)
      
def getSeverityDict():
  global serverityDictionary
  with open('MasterData/symptom_severity.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    try:
      for row in csv_reader:
        _diction = {row[0]: int(row[1])}
        serverityDictionary.update(_diction)
    except:
      pass
      
def getprecautionDict():
  global precautionDictionary
  with open('MasterData/symptom_precaution.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
      _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
      precautionDictionary.update(_prec)
      
def getInfo():
    return "------------------------- HealthCare ChatBot --------------------------\nYour Name?"


def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if (len(pred_list) > 0):
        return 1, pred_list
    else:
        return 0, []


def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


def tree_to_code(tree, feature_names):
    tree_ = tree.tree

@app.route('/', methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    # Get user input from form
    symptoms = request.form.get('symptoms')
    
    # Make prediction
    prediction, descriptions, precautions = predict_disease(symptoms)

    # Render template with results
    mytemplate = Template("<html><body><h1>HealthCare ChatBot</h1><p>Enter your symptoms </p><h2>Predicted Disease:</h2><p>${prediction}</p><h2>Description:</h2><p>${descriptions}</p><h2>Precautions:</h2><ul> {% for precaution in precautions %}<li> ${precaution} </li></ul></body></html>")
    return mytemplate.render(prediction=prediction, descriptions=descriptions, precautions=precautions)
  # Render empty form
  # return render_template('index.html')
  
  symptoms = request.form.get('symptoms')
  prediction, descriptions, precautions = predict_disease(symptoms)
  mytemplate = Template("<html><body><h1>HealthCare ChatBot</h1><p>Enter your symptoms </p><h2>Predicted Disease:</h2><p>${prediction}</p><h2>Description:</h2><p>${descriptions}</p><h2>Precautions:</h2><ul> {% for precaution in precautions %}<li> ${precaution} </li></ul></body></html>")
  return mytemplate.render(prediction=prediction, descriptions=descriptions, precautions=precautions)

  # return "Hello, world"
def predict_disease(symptoms):
  # Read in symptom severity, description, and precaution data
  getSeverityDict()
  getDescription()
  getprecautionDict()
  
  # Convert user input of symptoms to a list
  if symptoms is not None:
    symptoms_list = symptoms.split(',')
  else:
      symptoms_list = []

  # Convert symptom names to numbers
  symptom_indices = [symptoms_dict[symptom] for symptom in symptoms_list]

  # Create a vector with 0s and 1s indicating whether each symptom is present or not
  input_vector = np.zeros(len(symptoms_dict))
  input_vector[symptom_indices] = 1

  # Make prediction using decision tree
  prediction = print_disease(clf.predict([input_vector]))

  # Get descriptions and precautions for predicted disease
  descriptions = description_list[prediction[0]]
  precautions = precautionDictionary[prediction[0]]

  # Return prediction, descriptions, and precautions
  return prediction[0], descriptions, precautions

if __name__ == '__main__':
  app.run(debug=True)