import numpy as np
import streamlit as st
import pandas as pd 
import time
import matplotlib.pyplot as plt
import matplotlib
import json
import plotly.graph_objects as go
import urllib

from urllib.request import urlopen
from streamlit.report_thread import get_report_ctx
import requests


import configparser

# lecture des paramtres
config = configparser.ConfigParser()
config.read('config.ini')

URL       = config['config']['URL']
API       = config['config']['API']


st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 600px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 600px;
        margin-left: -600px;
    }

    /*
    .reportview-container .main .block-container{{
        max-width: 100%;
    */
 
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
 
    .css-12oz5g7 {
        flex: 1 1 0%;
        width: 100%;
        padding: 6rem 1rem 10rem;
        max-width: 100%;
    }

    .css-1d391kg {
        background-color: rgb(240, 242, 246);
        background-attachment: fixed;
        flex-shrink: 0;
        height: 100vh;
        overflow: auto;
        padding: 2rem 1rem;
        position: relative;
        transition: margin-left 300ms ease 0s, box-shadow 300ms ease 0s;
        width: 21rem;
        z-index: 100;
        /* margin-left: 0px; */
    }

    </style>
    """,
    unsafe_allow_html=True,
)


sb = st.sidebar 
sb.image(URL+'/P7/home.png')

# liste des clients 
response = urlopen(API+"/api/clients")
list_id = response.read().decode('utf-8')    
list_id = list_id.split(',')
numclient = sb.selectbox('Select a client ? (103625: non solvable/105091: solvable)', (list_id)) 

sb.image(URL+"/P7/feature_importance.png")

def newindictor(p_key):
    
    indicator_ = st.selectbox('Select a variable  ?',
                             ('','NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
                            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                            'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH'), key=p_key) 
    
    if (indicator_):

        response = urlopen(API+"/api/indicator?id="+str(numclient)+"&indic="+ str(indicator_))
        id = response.read().decode('utf-8')
        st.image(URL+'/P7/'+id+'.png')
        
        newindictor(p_key+"1")
        
    return indicator_

indicators = []

if (numclient !=""):

    col1, col2 = st.columns(2) 
    col3, col4 = st.columns(2) 

    response = urlopen(API+"/api/client/"+ str(numclient) )

    data_json = json.loads(response.read())
    score = float(data_json["score"])
    proba0 = float(data_json["proba0"])
    seuil = float(data_json["seuil"])
    json = data_json["json"]
    
    mondf = pd.read_json(json)
    sb.dataframe(mondf.T, 1000, 500)

    score = int(round(proba0,0))
    
    if score < seuil:
        color = "red"
        message = "Avis défavorable"
    else:
        color = "green"
        message = "Avis favorable"
        
    # gauge
    fig4 = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = score,
        mode = "gauge+number",
        
        title = {'text': message},
        delta = {'reference': 100},
        gauge = {'axis': {'range': [None, 100]},
             'steps' : [
                 {'range': [0, seuil], 'color': "red"},
                 {'range': [seuil, 100], 'color': "green"}],
             'bar': {'color': "lime"},
             'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 1, 'value': seuil}}))

    fig4.update_layout(font = {'color': color, 'family': "Arial"})
    fig4.show()
    col1.plotly_chart(fig4, use_container_width=True)
    
    col2.image(URL+'/P7/feature_importance_'+str(numclient)+'.png')
    
    #json = data_json["variables2"]
    #bbbb = data_json["variables_princ"]
    mondf = pd.read_json(data_json["variables2"])
    
    #aaaa=  list(mondf[0])
      
    indicator = col3.selectbox(' Variables explicatives', (list(mondf[0])), key="key1")         
    if (indicator):
        
        response = urlopen(API+"/api/indicator?id="+str(numclient)+"&indic="+ urllib.parse.quote(indicator))
        id = response.read().decode('utf-8')
        col3.image(URL+'/P7/'+id+'.png')

    col3.write("[nomenclature]("+ URL + "/P7/nomenclature.html)")
   
    indicator2 = col4.selectbox(' Variables importantes', (data_json["variables_princ"]), key="key2") 
    
    
    if (indicator2):
        
        response = urlopen(API+"/api/indicator?id="+str(numclient)+"&indic="+ urllib.parse.quote(indicator2))
        id = response.read().decode('utf-8')
        col4.image(URL+'/P7/'+id+'.png')
