#!/usr/bin/env python
import os
import os.path
import pandas as pd
import numpy as np
import streamlit as st
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
#import psycopg2 #(this broke streamlit)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


import joblib


####
'''
# Readmission Risk Estimator

This is some _markdown_.
'''

df = pd.DataFrame({'col1': [1,2,3]})
df  # <-- Draw the dataframe

x = 10
'x', x  # <-- Draw the string 'x' and then the value of x


#load the model and example data
X_example = pd.read_csv('readmission_example_patients.csv', index_col=0)
# loaded_model = pickle.load(open('rf_model.sav', 'rb'))
loaded_model = joblib.load(open('rf_model.sav', 'rb'))
probs_example = loaded_model.predict_proba(X_example)
scores_example = probs_example[:, 1]
patient_options = X_example.index.tolist()


#get patient selection 
patient = 'Jane Middleman'
patient = st.selectbox('Select patient:', patient_options)
psub = X_example.loc[patient,:]

#print the patient's data
psub


#predict probability from model
p_prob = loaded_model.predict_proba([psub])
p_prob = np.round(p_prob, 3)
st.write('Probability readmission = {}'.format(p_prob[0][1]))


#option to adjust the age
new_age = st.slider('adjust age', min_value = 18, max_value=91, format ='%d')
snew_age = float(scaler.fit_transform(np.array([18,new_age,91]).reshape(-1, 1))[1])

# #option to add bad hospital
# o96 = psub['hospitalid_hospital_96']
# ohospital = psub.loc[(psub.index.str.contains('hospitalid_hospital')) & (psub==1),].index
# change_96 = st.checkbox('hospital 96?')
# if change_96:
# 	psub.loc['hospitalid_hospital_96'] = 1.0
# 	psub.loc[ohospital] = 0.0
# else:
# 	psub.loc['hospitalid_hospital_96'] = 0.0
# 	psub.loc[ohospital] = 1.0

#get new probability
new_sub = psub.copy()
new_sub.loc[new_sub.index[0],'age'] = snew_age
new_prob = loaded_model.predict_proba([new_sub])
new_prob = np.round(new_prob, 3)
st.write('Adjusted probability = {}'.format(new_prob[0][1]))