#!/usr/bin/env python

#load functions
exec(open("functions.py").read())

#print the title
image = Image.open('data/logo.png')
st.image(image, format='PNG')



#--------------- selecte app subset
 #get patient selection from user
app_parts = ['Administrator overview', 'Discharge advisor']
current_part = st.sidebar.selectbox('Application:', app_parts)


#--------------- ADMINISTRATIVE OVERVIEW
if current_part == 'Administrator overview':
	st.markdown('## Administrator Overview', unsafe_allow_html=False)
	admin_options = ['Readmission times',
	                 'Feature importances',
	                 'Reduction goals']
	current_option = st.sidebar.selectbox('Overview options:', admin_options)
	if current_option == 'Readmission times':
		plot_readmission_hist()
	elif current_option == 'Feature importances':
		plot_importance_bars()
	elif current_option == 'Reduction goals':
		st.write('reduction goals')
		pr_dat, roc_dat, auc = load_performance()
		select_stat = st.selectbox('Target performance statistic', ['precision', 'recall'])
		if select_stat=='precision':
			select_precision = st.slider('Select target precision', min_value=0.0, max_value=max(pr_dat['Precision']), value=0.16)
			target_threshold, recall_result = plot_target_precision(select_precision)
			st.write('The selected precision is {}.'.format(select_precision))
			st.write('The resulting recall is {}.'.format(recall_result))
		elif select_stat=='recall':
			select_recall = st.slider('Select target recall', min_value=0.0, max_value=max(pr_dat['Recall']), value=0.16)
			target_threshold, precision_result = plot_target_recall(select_recall)
			st.write('The selected recall is {}.'.format(select_recall))
			st.write('The resulting precision is {}.'.format(precision_result))


#--------------- DISCHARGE ADVISOR
elif current_part == 'Discharge advisor':

	#--------------- recomendation part
	#load the model and example data and the model
	X_example = pd.read_csv('data/readmission_example_patients.csv', index_col=0)
	loaded_model = joblib.load(open('data/rf_model.joblib', 'rb'))

	#get the example patient names
	patient_options = X_example.index.tolist()

	#get patient selection from user
	patient = st.sidebar.selectbox('Select patient:', patient_options)
	psub = X_example.loc[patient,:] #load data for prediction


	#activate
	#predict probability from model
	p_prob = loaded_model.predict_proba([psub])[0,1]
	p_prob = np.round(p_prob, 3)
	readmission_size = p_prob*100
	stay_home_size = 100.0 - readmission_size



	# #plot probability
	pcts = [readmission_size, stay_home_size]
	names = ['readmission', 'safe']
	df = pd.DataFrame({'patient':patient, 'percentage':pcts, 'prediction':names})
	fig = px.bar(df, x="percentage", y="patient", color='prediction', orientation='h',
	             height=300,
	             hover_data=["prediction", "percentage"],
	             color_discrete_map={'readmission':'firebrick',
	                                  'safe':'grey'})
	fig.update_layout(
	    title='Probability of readmission within 30 days of discharge',
	    yaxis_title="",
	    plot_bgcolor='rgba(0,0,0,0)')
	st.plotly_chart(fig, use_container_width=True)



	#------- PLOT SUGGESTION -------#

	#images
	image_paths = {
	'very_high': 'data/stop.png',
	'high': 'data/stop.png',
	'medium': 'data/caution.png',
	'low': 'data/ok.png'
	}

	#cpations
	captions = {
	'very_high':
	'This patient is at very high risk of readmission within 30 days. Dischage strongly discouraged.',
	'high':
	'This patient is at high risk of readmission within 30 days. Dischage discouraged.',
	'medium':
	'This patient posses some risk for readmission within 30 days. Dischage with caution.',
	'low':
	'This patient is of low concern for readmission.'}

	#function to plot
	def plot_recommendation_image(image_path, caption, image_width):
		rec_image = Image.open(image_path)
		st.image(rec_image, width = image_width, format='PNG')
		st.write(report)
		st.write(caption)

	report = 'The probability that the {} will be readmitted within 30 days is {}%'.format(patient,p_prob*100)
	image_width = 80
	if p_prob > 0.75:
		plot_recommendation_image(image_paths['very_high'], captions['very_high'], image_width)
	elif p_prob > 0.49:
		plot_recommendation_image(image_paths['high'], captions['high'], image_width)
	elif p_prob > 0.1:
		plot_recommendation_image(image_paths['medium'], captions['medium'], image_width)
	else:
		plot_recommendation_image(image_paths['low'], captions['low'], image_width)
		pass


	#------- SHOW PATIENT DATA -------#

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

	#subset for patient
	psub_view = X_view.loc[patient,:] #load data for viewing

	#print the patient's data to streamlit screen
	'''
	### patient data:
	'''
	st.write(psub_view)