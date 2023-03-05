#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[43]:


import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image


# In[44]:


data_test = pd.read_csv('C:/Users/AB54/Desktop/streamlit_project/heart/test.csv')


# In[45]:


data_test['age'] = data_test['age'].astype(float)
data_test['age'] = round(data_test['age']/365, 2)
data_test['smoke'] = data_test['smoke'].astype(bool)
data_test['alco'] = data_test['alco'].astype(bool)
data_test['active'] = data_test['active'].astype(bool)


# In[46]:



pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)
  
def welcome():
    return 'welcome all'
  

def prediction(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):  
   
    prediction = classifier.predict(
        [[age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
    print(prediction)
    return prediction
      

def main():
      # giving the webpage a title
    st.title("Heart diseases")
      
    
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">RandomForestClassifier ML App </h1>
    </div>
    
     """ 
  
    st.markdown(html_temp, unsafe_allow_html = True)
      
    
    age = st.number_input("age",
                          min_value=float(data_test['age'].min()),
    max_value=float(data_test['age'].max()))
    st.write('The age is ', age)
        
    gender = st.radio("gender", data_test['gender'].unique())
    st.write("gender:", gender)
   
   
    height = st.number_input("height",
                          min_value=float(data_test['height'].min()),
    max_value=float(data_test['height'].max()))
    st.write('The height is ', age)
    
    weight = st.number_input("weight",
                          min_value=float(data_test['weight'].min()),
    max_value=float(data_test['weight'].max()))
    st.write('The weight is ', age)
    
    ap_columns = st.columns(2)
    ap_hi = ap_columns[0].number_input("ap_hi", min_value=float(data_test['ap_hi'].min()),
    max_value=float(data_test['ap_hi'].max()))
    ap_lo = ap_columns[1].number_input("ap_lo", min_value=float(data_test['ap_lo'].min()),
    max_value=float(data_test['ap_lo'].max()))
    if ap_hi < ap_lo:
        st.error("The ap_hi can't be smaller than the ap_lo!")
    else:
        st.success("Congratulations! Correct Parameters!")
        subset_ap = data_test[(data_test['age'] <= ap_hi) & (ap_lo <= data_test['age'])]
        st.write(f"ap is {ap_hi} and {ap_lo}")
        
        
   
    
    cholesterol = st.radio("cholesterol", data_test['cholesterol'].unique())
    st.write("cholesterol:", cholesterol)
    
    gluc = st.radio("gluc", data_test['gluc'].unique())
    st.write("gluc:", gluc)
    
    smoke = st.selectbox("Dou you smoke?", data_test['smoke'].unique())
    st.write(f"Selected Option: {smoke!r}")

     
    alco = st.selectbox("What about alco?", data_test['alco'].unique())
    st.write(f"Selected Option: {alco!r}")
    
    active = st.selectbox("What about sport?", data_test['active'].unique())
    st.write(f"Selected Option: {active!r}")
    
    
    result =""
      
    
    if st.button("Predict"):
        result = prediction(age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active)
    st.success('The output is {}'.format(result))
     
if __name__=='__main__':
    main()


# In[ ]:





# In[ ]:




