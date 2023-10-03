import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib
from tensorflow import keras
from test import res
app=Flask(__name__)
data=pd.read_csv('TRAVEL.csv')
#model=pickle.load(open("Sid.pkl",'rb'))
#model = keras.models.load_model('travel_insurance_model.h5')
@app.route('/')
def index():
    agency_list = sorted(data['Agency'].unique())
    agency_type_list = sorted(data['Agency Type'].unique())
    distribution_channels = sorted(data['Distribution Channel'].unique())
    product_names = sorted(data['Product Name'].unique())
    d_names = sorted(data['Destination'].unique())
    return render_template('index.html', agencies=agency_list, agency_type=agency_type_list, ch=distribution_channels, pn=product_names,gender=['M','F',None],d=d_names)

@app.route('/predict', methods=['POST'])
def predict():
    l=[]
    Agency = request.form.get('agency')
    Agency_Type=request.form.get('t')
    dis=request.form.get('s')
    pn=request.form.get('product')
    d=request.form.get('duration')
    des=request.form.get('destination')
    g=request.form.get('gender')
    a=request.form.get('age')
    #print(location, bhk, bath,sqft)
    input = pd.DataFrame([[Agency,Agency_Type,dis,pn,d,des,g,a]],columns=['Agency','Agency Type','Distribution Channel','Product Name','Duration','Destination','Gender','Age'])
    l.extend([Agency,Agency_Type,dis,pn,d,des,g,a])
    return res(l)

'''
def predict1():
    Agency = request.form.get('agency')
    Agency_Type=request.form.get('t')
    dis=request.form.get('s')
    pn=request.form.get('product')
    d=request.form.get('duration')
    des=request.form.get('destination')
    g=request.form.get('gender')
    a=request.form.get('age')
    #print(location, bhk, bath,sqft)
    input = pd.DataFrame([[Agency,Agency_Type,dis,pn,d,des,g,a]],columns=['Agency','Agency Type','Distribution Channel','Product Name','Duration','Destination','Gender','Age'])
    prediction=model.predict(input)[0] * 1e5
    return ""
    #return str(np.round(prediction,2))
'''

if __name__=="__main__":
    app.run(debug=True, port=5001)
