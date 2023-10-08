from flask import Flask, render_template, request, redirect, url_for
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import pickle
import sys

scaler=pickle.load(open("/my fire model all /models/fire_scaler.plk","rb"))
linear=pickle.load(open("/my fire model all /models/fire_linear.plk","rb"))

column_names=list(scaler.feature_names_in_)
# print(column_names) #hello update

app = Flask(__name__)

@app.route('/home', methods=['GET', 'POST'])
def run_python_function():
    result="NONE"
    if request.method == 'POST':
        user_inputs = {}
        user_list_data=[]
        print('Hello world!', file=sys.stderr)
        # print(user_inputs)
        for column in column_names:
            value = request.form.get(column)
            user_inputs[column] = value
            user_list_data.append(float(value))
        print(user_list_data)
        scaled_data=scaler.transform([user_list_data])
        result=linear.predict(scaled_data)
        return render_template('result.html', results=user_inputs,result=result) 
    return render_template('index.html', column_names=column_names)

if __name__ == '__main__':
    app.run(debug=True,port=2112)
    