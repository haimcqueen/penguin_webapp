import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

st.write("""
# Penguin Prediction App

This app predicts the **Palmer Penguin** species!

**Data:** [palmer penguins library]("https://github.com/allisonhorst/palmerpenguins/") by Allison Horst in R

""")
example_input = pd.read_csv('model_building/data/penguins_example.csv')

st.sidebar.header('User Input Features')
st.markdown("""
[Example CSV input file](https://github.com/dataprofessor/data/blob/master/penguins_example.csv)
""")
example_input

uploaded_file = st.sidebar.file_uploader("Upload your csv file", type=['csv'])

if uploaded_file is not None:
    input_df = pd.csv_read(uploaded_file)

else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill Length (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()

# concat user input with penguins dataset for encoding
penguins_raw = pd.read_csv('model_building/data/penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

target = 'species'
encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]

# Display user input
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file upload. Currently using example input parameters (shown below).')
    st.write(df)

with open('model_building/penguin_clf.pkl', 'rb') as model_file:
    load_clf = pickle.load(model_file)

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Predition')
penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
