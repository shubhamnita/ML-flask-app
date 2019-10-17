import streamlit as st
import pandas as pd
import numpy as np

st.title('Kaggle Titanic Dataset')

# Load the dataset in a dataframe object and include only four features as mentioned

DATA_URL = ('http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv')

def main():
	data_load_state = st.text('Loading data...')
	data = load_data(1000)
	data_load_state.text('Loading data... done!')
	if st.checkbox('Show raw data'):
		st.subheader('Raw data')
		st.write(data)
	st.subheader('Feature Columns Selection')
	features = st.multiselect('Feature-Columns',data.columns)
	st.write(features)
	target = st.selectbox('Target Column',data.columns)


@st.cache
def load_data(nrows):
	data = pd.read_csv(DATA_URL, nrows=nrows)
	lowercase = lambda x: str(x).lower()
	data.rename(lowercase, axis='columns', inplace=True)
	return data

if __name__ == "__main__":
    main()