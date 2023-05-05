import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)
def prediction(m,island,bill_l,bill_d,flipper_l,body_mass,sex):
  species = m.predict([[island,bill_l,bill_d,flipper_l,body_mass,sex]])
  species = species[0]
  if species == 0:
    return 'Adelie'
  elif species == 1:
    return 'Chinstrap'
  else:
    return 'Gentoo'
# Design the App
st.sidebar.title('Idenitifiying the penguin')
l = st.sidebar.slider('Bill_length_mm',bill_length_mm.min(),bill_length_mm.max())
d=st.sidebar.slider('bill_depth_mm',bill_depth_mm.min(),bill_depth_mm.max())
fl=st.sidebar.slider('flipper_length_mm',flipper_length_mm.min(),flipper_length_mm.max())
mas = st.sidebar.slider('body_mass_g',body_mass_g.min(),body_mass_g.max())
sex = st.sidebar.selectbox('Sex',('Male','Female'))
island = st.sidebar.selectbox('Island',('Biscoe', 'Dream', 'Torgersen'))
cla = st.sidebar.selectbox('Classifier',('SVC','RFC','LR'))
if st.sidebar.button('Predict'):
  if cla == 'SVC':
    p = prediction(svc_model,island,l,d,fl,mas,sex)
    st.write('The name of the species:',p)
    st.write('The accuracy of the model is :',svc_score)
  elif cla == 'RFC':
    q = prediction(rf_clf,island,l,d,fl,mas,sex)
    st.write('The name of the species:',q)
    st.write('The accuracy of the model is :',rf_clf_score)
  else:
    r = prediction(log_reg,island,l,d,fl,mas,sex)
    st.write('The name of the species:',r)
    st.write('The accuracy of the model is :',log_reg_score)