#!/usr/bin/env python
import os
import os.path
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from PIL import Image

#print the title
image = Image.open('data/logo.png')
st.image(image, format='PNG')


#format the table to look prettier
X_view = pd.read_csv('data/readmission_example_patients_view.csv', index_col=0)
col_change = {'INSURANCE':'insurance type',
'RELIGION': 'religion', 
'MARITAL_STATUS':'marital status',
'hospital_duration':'admission duration',
'ADMISSION_LOCATION':'admission location',
'DISCHARGE_LOCATION':'discharge location',
'GENDER': 'gender',
'age':'age',
'nicu_stays':'total ICU visits',
'total_icu_days':'total days in ICU',
'total_diagnoses':'total diagnoses',
'total_procedure':'total procedures',
'total_drug':'total drugs prescribed'}
X_view = X_view.rename(columns = col_change)
X_view = X_view.round(decimals=0)
num_cols = ['admission duration',
'age',
'total ICU visits',
'total days in ICU',
'total drugs prescribed']
X_view[num_cols] = X_view[num_cols].astype('int')


#load the model and example data and teh model
X_example = pd.read_csv('data/readmission_example_patients.csv', index_col=0)
loaded_model = joblib.load(open('data/rf_model.joblib', 'rb'))

#get the example patient names
patient_options = X_example.index.tolist()


#get patient selection from user
patient = st.sidebar.selectbox('Select patient:', patient_options)
psub = X_example.loc[patient,:] #load data for prediction
psub_view = X_view.loc[patient,:] #load data for viewing


#predict probability from model
p_prob = loaded_model.predict_proba([psub])[0,1]
p_prob = np.round(p_prob, 3)
readmission_size = p_prob*100
stay_home_size = 100.0 - readmission_size



# #plot probability
pcts = [stay_home_size, readmission_size]
names = ['safe', 'readmission risk']
df = pd.DataFrame({'pcts':pcts, 'names':names})
fig = px.pie(df, values='pcts', names='names',
             color='names',
             color_discrete_map={'readmission risk':'firebrick',
                                 'safe':'darkgrey'})
fig.update_layout(showlegend=True)
st.plotly_chart(fig, use_container_width=True)


#plot recommendation symbol
report = 'The probability that the {} will be readmitted within 30 days is {}'.format(patient,p_prob)
if p_prob > 0.75:
	rec_image = Image.open('data/stop.png')
	st.image(rec_image, format='PNG')
	st.write(report)
	st.write('This patient is at very high risk of readmission within 30 days. Confirmation with physician required')
elif p_prob > 0.49:
	rec_image = Image.open('data/stop.png')
	st.image(rec_image, format='PNG')
	st.write(report)
	st.write('This patient is at high risk of readmission within 30 days. Suggest confirmation with physician')
elif p_prob > 0.1:
	rec_image = Image.open('data/caution.png')
	st.image(rec_image, format='PNG')
	st.write(report)
	st.write('This patient posses some risk for readmission within 30 days and should be discharged with caution.')
else:
	rec_image = Image.open('data/ok.png')
	st.image(rec_image, format='PNG')
	st.write(report)
	st.write('This patient is of low concern for readmission.')
	pass

#print the patient's data to streamlit screen
'''
### patient data:
'''
st.write(psub_view)
