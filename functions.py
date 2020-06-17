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


def plot_readmission_hist():
	read_dat = joblib.load(open('/Users/grovesdixon/gitreps/Insight_fellowship_project/data/for_app/read_dat.joblib', 'rb'))
	fig = px.histogram(read_dat.loc[read_dat['day']<365,:], x="day", nbins=365,
                  color_discrete_sequence=['grey'])
	fig.update_layout(title='Readmissions',
	                  plot_bgcolor='rgba(0,0,0,0)')
	st.plotly_chart(fig, use_container_width=True)
