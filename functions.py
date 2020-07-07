#!/usr/bin/env python
import os
import os.path
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


def plot_readmission_hist():
	dat = joblib.load(open('data/read_dat.joblib', 'rb'))
	fig = px.histogram(dat.loc[dat['day']<365,:], x="day", nbins=365,
                  color_discrete_sequence=['grey'])
	fig.update_layout(title='Readmissions', xaxis_title="Days between discharge and readmission",
					  yaxis_title='Readmission count',
	                  plot_bgcolor='rgba(0,0,0,0)')
	st.plotly_chart(fig, use_container_width=True)


def plot_importance_bars():
	dat = joblib.load(open('data/importances.joblib', 'rb'))
	dat = dat.loc[0:49,:]
	# dat['feature'] = dat['feature'].str.replace('DNword_','Discharge Notes: ')
	# dat['feature'] = dat['feature'].str.replace('drug_','Drug: ')
	fig = go.Figure()
	fig.add_trace(go.Bar(
	    name='Control',
	    x=dat['feature'], y=dat['importance'],
	    marker_color='grey',
	    #error_y=dict(type='data', array=dat['std'], thickness=1)
	))
	fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
					  yaxis_title='Feature importance',
					  autosize=False,
					    width=600,
					    height=700,
					    margin=dict(
					        l=50,
					        r=50,
					        b=300,
					        t=100,
					        pad=4
					)
				    )
	st.plotly_chart(fig, use_container_width=False)
	#st.write('(enter full screen mode to see feature labels)')


def load_performance():
	test_perf = joblib.load(open('data/performance.joblib', 'rb'))
	auc = test_perf['auc']
	#format precision and recall data
	pr_dat = pd.DataFrame({'Recall':test_perf['recall'],
	                         'Precision':test_perf['precision'],
	                         'threshold':test_perf['recall_prec_thresholds']})
	pr_long = pd.melt(pr_dat, id_vars=['threshold'], value_vars=['Recall', 'Precision'], var_name='Stat')

	#format ROC data
	roc_dat = pd.DataFrame({'False Positive Rate':test_perf['fpr'],
	                         'True Positive Rate':test_perf['tpr'],
	                         'threshold':test_perf['roc_thresholds']})
	tot = roc_dat['False Positive Rate'] + roc_dat['False Positive Rate']
	roc_dat = roc_dat.loc[tot>0,:]
	return pr_dat, roc_dat, auc

def plot_target_precision(select_precision=0.16):
	#PLOT WITH SELECTED PRECISION
	pr_dat['dist'] = abs(pr_dat['Precision']-select_precision)
	pr_sub1 = pr_dat.loc[pr_dat['dist']==min(pr_dat['dist']),:]
	pr_sub2 = pr_sub1.loc[pr_sub1['Recall']==max(pr_sub1['Recall']),]
	pr_sub3 = pr_sub2.loc[pr_sub2['threshold']==max(pr_sub2['threshold']),]
	print(pr_sub1)
	print(pr_sub2)
	target_recall = float(pr_sub3['Recall'])
	target_threshold = float(pr_sub3['threshold'])
	target_precision = float(pr_sub3['Precision'])


	# Create traces
	fig = go.Figure()
	#add precision trace
	fig.add_trace(go.Scatter(x=pr_dat['threshold'], y=pr_dat['Precision'],
	                    mode='lines',
	                    name='Precision', marker_color='firebrick'))
	#add recall trace
	fig.add_trace(go.Scatter(x=pr_dat['threshold'], y=pr_dat['Recall'],
	                    mode='lines',
	                    name='Recall', marker_color='grey'))
	#add vertical intersect line
	fig.add_shape(type="line",
	            x0=target_threshold,
	            y0=0,
	            x1=target_threshold,
	            y1=target_recall,
	            line=dict(
	                color="grey",
	                width=1,
	                dash="dashdot"))
	#add horizontal intersect line
	fig.add_shape(
	            type="line",
	            x0=0,
	            y0=target_recall,
	            x1=target_threshold,
	            y1=target_recall,
	            line=dict(
	                color="grey",
	                width=1,
	                dash="dashdot"))
	fig.add_trace(go.Scatter(
	    x=[target_threshold],
	    y=[target_precision],
	    mode="markers",
	    name="Precision target = {}".format(round(select_precision, 3)),
	    marker=dict(size=[12],
	               color = ['firebrick'])
	))
	fig.add_trace(go.Scatter(
	    x=[target_threshold],
	    y=[target_recall],
	    mode="markers",
	    name="Recall result = {}".format(round(target_recall, 3)),
	    marker_color='grey',
	    marker=dict(size=[12])
	))
	fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
					 title = 'Precision-recall curves',
	                 yaxis_title="Stat value",
	                 xaxis_title='Readmission probability threshold')
	fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black')
	fig.update_yaxes(showline=True, linewidth=1.5, linecolor='black')
	st.plotly_chart(fig, use_container_width=True)
	return round(target_threshold,3), round(target_recall, 3)



def plot_target_recall(select_recall=0.16):
	pr_dat['dist'] = abs(pr_dat['Recall']-select_recall)
	pr_sub1 = pr_dat.loc[pr_dat['dist']==min(pr_dat['dist']),:]
	pr_sub2 = pr_sub1.loc[pr_sub1['Precision']==max(pr_sub1['Precision']),]
	pr_sub3 = pr_sub2.loc[pr_sub2['threshold']==max(pr_sub2['threshold']),]
	print(pr_sub1)
	print(pr_sub2)
	print(pr_sub3)
	target_recall = float(pr_sub3['Recall'])
	target_threshold = float(pr_sub3['threshold'])
	target_precision = float(pr_sub3['Precision'])


	# Create traces
	fig = go.Figure()
	#add precision trace
	fig.add_trace(go.Scatter(x=pr_dat['threshold'], y=pr_dat['Precision'],
	                    mode='lines',
	                    name='Precision', marker_color='firebrick'))
	#add recall trace
	fig.add_trace(go.Scatter(x=pr_dat['threshold'], y=pr_dat['Recall'],
	                    mode='lines',
	                    name='Recall', marker_color='grey'))
	#add vertical intersect line
	fig.add_shape(
	            type="line",
	            x0=target_threshold,
	            y0=0,
	            x1=target_threshold,
	            y1=target_recall,
	            line=dict(
	                color="grey",
	                width=1,
	                dash="dashdot"))
	#add horizontal intersect line
	fig.add_shape(
	            type="line",
	            x0=0,
	            y0=target_recall,
	            x1=target_threshold,
	            y1=target_recall,
	            line=dict(
	                color="grey",
	                width=1,
	                dash="dashdot"))
	fig.add_trace(go.Scatter(
	    x=[target_threshold],
	    y=[target_precision],
	    mode="markers",
	    name="Precision result = {}".format(round(target_precision, 3)),
	    marker=dict(size=[12],
	               color = ['firebrick'])
	))
	fig.add_trace(go.Scatter(
	    x=[target_threshold],
	    y=[target_recall],
	    mode="markers",
	    name="Recall target = {}".format(round(select_recall, 3)),
	    marker_color='grey',
	    marker=dict(size=[12])
	))
	fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
	                 yaxis_title="Stat value",
	                 xaxis_title='Readmission probability threshold')
	fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black')
	fig.update_yaxes(showline=True, linewidth=1.5, linecolor='black')
	st.plotly_chart(fig, use_container_width=True)
	return round(target_threshold,3), round(target_precision, 3)
