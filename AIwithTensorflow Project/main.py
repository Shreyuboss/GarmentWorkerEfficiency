import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

# Load the saved Boosting Regressor model
model_path = 'C:/Users/grshr/OneDrive/Desktop/AIwithTensorflow Project/productivity.pkl'
model = pickle.load(open(model_path, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/data_predict', methods=['POST'])
def data_predict():
    quarter = int(request.form['Quarter'])
    department = request.form['Department']
    if department == 'Sewing':
        department = 1
    elif department == 'Finishing':
        department = 0
    
    day = request.form['Day']
    if day == 'Monday':
        day = 0
    elif day == 'Tuesday':
        day = 1
    elif day == 'Wednesday':
        day = 2
    elif day == 'Thursday':
        day = 3
    elif day == 'Friday':
        day = 4
    elif day == 'Saturday':
        day = 5
    elif day == 'Sunday':
        day = 6
    
    team_number = int(request.form['Team Number'])
    time_allocated = int(request.form['Time Allocated'])
    unfinished_items = int(request.form['Unfinished Items'])
    over_time = int(request.form['Over time'])
    incentive = int(request.form['Incentive'])
    idle_time = int(request.form['Idle Time'])
    idle_men = int(request.form['Idle Men'])
    style_change = int(request.form['style Change'])
    no_of_workers = int(request.form['Number of Workers'])
    
    prediction = model.predict(pd.DataFrame([[quarter, department, day, team_number, time_allocated, 
                                              unfinished_items, over_time, incentive, idle_time, 
                                              idle_men, style_change, no_of_workers]]))
    
    prediction = np.round(prediction, 4) * 100
    prediction_text = f"The predicted productivity is: {prediction}%"
    
    return render_template('productivity.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
