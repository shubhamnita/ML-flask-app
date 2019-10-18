import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.title('Who survived Titanic - Jake or Rose')

# Load the dataset in a dataframe object and include only four features as mentioned

DATA_URL = ('http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv')

def main():
	data_load_state = st.text('Loading data...')
	data = load_data(1000)
	data_load_state.text('Loading data... done!')
	st.sidebar.header('Options')
	if st.sidebar.checkbox('Show raw data'):
		st.subheader('Kaggle Titanic Dataset')
		st.write(data)
	'''	
	## Feature and Target Columns Selection
	'''
	feature_columns = st.sidebar.multiselect('Feature-Columns',data.columns)
	st.write(feature_columns)
	dependent_variable = st.sidebar.selectbox('Target Column',data.columns)
	include = feature_columns
	include.append(dependent_variable)
	df_ = data[include]
	if st.sidebar.checkbox('Show filtered data'):
		st.subheader('Filtered data')
		st.write(df_)
		
	# Data Preprocessing
	st.subheader('Data Preprocessing and Cleaning Steps')
	st.text('Null Values information')
	st.write(df_.isnull().sum())
	na_remove_state = st.text('Removing Null values...')
	df_.dropna(axis=0,inplace=True)
	na_remove_state.text('Removed Null values... done!')
	if st.sidebar.checkbox('Show filtered data after null values removal'):
		st.subheader('Filtered data')
		st.write(df_)

	categoricals = []
	for col, col_type in df_.dtypes.iteritems():
		if col_type == 'O':
			categoricals.append(col)
		else:
			df_[col].fillna(0, inplace=True)

	df_ohe = pd.get_dummies(df_, columns=categoricals)
	if st.sidebar.checkbox('Show filtered data after handling categorical data'):
		st.subheader('Filtered data')
		st.write(df_ohe)

	if st.sidebar.checkbox('Train the classification model'):
		p_holder = st.empty()
		algorithm = st.selectbox('Classification Algorithm',['LogisticRegression'])
		dependent_variable = 'survived'
		if dependent_variable in df_ohe.columns:
			x = df_ohe[df_ohe.columns.difference([dependent_variable])]
			y = df_ohe[dependent_variable]
			model = train_model(x,y)

	if st.sidebar.checkbox('Want to predict'):
		st.subheader('Enter Test data')
		test_data ={}
		for var in df_ohe.columns.difference([dependent_variable]):
			
			test_data[var] = int(st.text_input(var),10)
			

		#st.write(test_data)
		#test_df = pd.DataFrame(test_data,columns=df_ohe.columns.difference([dependent_variable]))
		st.subheader('Test data is: ')
		test_df = pd.DataFrame([test_data])
		st.write(test_df)
		prediction = list(model.predict(test_df))
		st.subheader('Prediction is: ')
		st.write(str(prediction[0]))
				


@st.cache
def load_data(nrows):
	data = pd.read_csv(DATA_URL, nrows=nrows)
	lowercase = lambda x: str(x).lower()
	data.rename(lowercase, axis='columns', inplace=True)
	return data

def train_model(x,y):
	# Logistic Regression classifier
	lr = LogisticRegression()
	lr.fit(x, y)
	return lr

if __name__ == "__main__":
    main()