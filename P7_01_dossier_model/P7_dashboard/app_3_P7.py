import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
import plotly.graph_objects as go
import urllib
import pickle5 as pickle
import seaborn as sns
import shap
import dill
from urllib.request import urlopen
import requests
import configparser

seuil = 0.5752224859896156

PATH_PICKLE = '../P7_api/pickle/'
URL = "http://127.0.0.1"
API = "http://127.0.0.1:5000"

# chargemet de la base X
X = pickle.load(open(PATH_PICKLE+'X.pickle', 'rb'))
# chargemet de la base X_train
X_train = pickle.load(open(PATH_PICKLE+'X_train.pickle', 'rb'))
# chargemet du pickle du modèle
model = pickle.load(open(PATH_PICKLE+'model.pkl', 'rb'))
# chargement de la base train_2
train_2 = pickle.load(open(PATH_PICKLE+'train_2.pickle', 'rb'))
variables = train_2.drop(columns=['index', 'SK_ID_CURR', 'TARGET'])
variable_list = variables.columns
id_client = train_2['SK_ID_CURR']
df_id_client = pd.DataFrame(id_client)



st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
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
with sb:
    st.image("./P7/pretadepenser.jpg")

# liste des clients
response = urlopen(API + "/api/clients")
list_id = response.read().decode('utf-8')
list_id = list_id.split(',')

numclient = sb.selectbox('Select a client ? (103625: non solvable/105091: solvable)', (list_id))

#with sb:

#    st.image("./feature_importance.png")

sb = st.sidebar

if (numclient != ""):

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)
    #col7, col8 = st.columns(2)

    response = urlopen(API + "/api/client/" + str(numclient))

    data_json = json.loads(response.read())
    score = float(data_json["score"])
    proba0 = float(data_json["proba0"])
    seuil = float(data_json["seuil"])
    json = data_json["json"]
    neighbors = data_json["json_1"]



    mondf = pd.read_json(json)
    sb.dataframe(mondf.T, 1000, 500)

    score = int(round(proba0, 0))

    if score < seuil:
        color = "red"
        message = "Avis défavorable"
    else:
        color = "green"
        message = "Avis favorable"

    #####
    ab = seuil-score
    #delta={'reference': 200},
    if ab >= 0:
        color = "red"
        profit = "défavorable"
    else:
        color = "green"
        profit = "Avis favorable"

    import plotly.graph_objects as go
############################################
    with col1:
        st.header(profit)
        st.subheader('This is a subheader')
        st.text('This is some text.')
        fig = go.Figure(go.Indicator(
            mode="number+gauge+delta", value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            delta={'reference': seuil, 'position': "top"},
            title={'text': "<b>profit</b><br><span style='color: gray; font-size:0.8em'>U.S. $</span>",
                   'font': {"size": 14}},
            gauge={
                'shape': "bullet",
                'axis': {'range': [None, 100]},
                'threshold': {
                    'line': {'color': "red", 'width': 2},
                    'thickness': 0.75, 'value': score},
                'bgcolor': "white",
                'steps': [
                    {'range': [0, score], 'color': "cyan"},
                    {'range': [0, seuil], 'color': "orangered"},
                    {'range': [150, score], 'color': "royalblue"}],
                'bar': {'color': "darkblue"}}))
        fig.add_annotation(x=0, y=2,
                           text="Text annotation with arrow",
                           showarrow=False,
                           arrowhead=1)
        fig.add_annotation(x=0, y=1.25,
                           text="Text annotation without arrow",
                           showarrow=False,
                           yshift=10)
        fig.update_layout(height=250)
        col1.plotly_chart(fig, use_container_width=True)

###############################
    with col2:

        st.image("./feature_importance.png")

 #################################
    with col3:
        with open(PATH_PICKLE + 'lime_.pickle', 'rb') as f: lime1 = dill.load(f)
        train_scale = train_2.drop(columns=['TARGET', 'index', 'SK_ID_CURR'])
        from sklearn.preprocessing import MinMaxScaler
        minmax_scale = MinMaxScaler()
        X_minmax = minmax_scale.fit_transform(train_scale)
        X_minmax = pd.DataFrame(X_minmax)
        X_minmax['SK_ID_CURR'] = train_2['SK_ID_CURR']

        train_5 = X_minmax.loc[X_minmax['SK_ID_CURR'] == int(numclient)]
        train_5 = train_5.drop(columns=['SK_ID_CURR'])
        train_5 = pd.DataFrame(train_5.values).iloc[0]
        #print(train_4)
        #print(pd.DataFrame(train_4.values).iloc[0])
        #train_5 = pd.DataFrame(train_5.values).iloc[0]
        #print(pd.DataFrame(X_train).iloc[int(numclient)])

        #train_4 = pd.DataFrame(train_4.values).iloc[0]
        exp = lime1.explain_instance(train_5,
                                     model.predict_proba,
                                     num_samples=100)

        exp.show_in_notebook(show_table=True)
        fig = exp.as_pyplot_figure()
        #plt.tight_layout()
        # a completer l'id client pur le vrai
        fig.savefig('feature_importance_5'+'.png', bbox_inches='tight')
        st.image('./feature_importance_5.png')

######################################
with col4:
    variable_explicative = st.selectbox('choisir une variable explicative)', (variable_list), key="lower")
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % variable_explicative)

    fig = sns.displot(variables[variable_explicative],color=color, kde=True,bins=100)


    plt.axvline(x=list(train_2[variable_explicative].loc[train_2['SK_ID_CURR'] == int(numclient)])[0], color='orange', label='Position du client')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    #fig = train_2[variable_explicative].hist()
    #plt.savefig('test_'+'.png')
    #st.image("./test_.png")

##############################################

with col5:
    target_line = st.selectbox('choisir une variable explicative)', (variable_list))
    plt.figure(figsize=(10,6))
    plt.title("Distribution of %s" % "target line")

    fig = sns.kdeplot(train_2.loc[train_2['TARGET'] == 0, target_line], label = 'target == 0')
    fig = sns.kdeplot(train_2.loc[train_2['TARGET'] == 1, target_line], label='target == 1')

    plt.axvline(x=list(train_2[variable_explicative].loc[train_2['SK_ID_CURR'] == int(numclient)])[0], color='orange', label='Position du client')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=18)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


    with col6:

        autre_clients = st.selectbox('Comparons notre client avec un autre client solvable ou pas solavble)', (list_id))
        ### Shape
        explainer = shap.TreeExplainer(model)
        #print(train_2)
        choosen_instance = train_2.loc[train_2['SK_ID_CURR']==int(autre_clients)]
        choosen_instance = choosen_instance.drop(columns=['TARGET', 'index', 'SK_ID_CURR'])
        #train_instance = train_2.drop(columns=['TARGET','index','SK_ID_CURR'])
        shap_values = explainer.shap_values(choosen_instance)
        shap.summary_plot(shap_values, choosen_instance)

        plt.savefig('feature_importance_1'+'.png', bbox_inches='tight')
        plt.tight_layout()
        #st.image('./feature_importance_1.png')

        # lime
        with open(PATH_PICKLE + 'lime_.pickle', 'rb') as f: lime1 = dill.load(f)
        #####
        train_scale = train_2.drop(columns=['TARGET', 'index', 'SK_ID_CURR'])
        from sklearn.preprocessing import MinMaxScaler
        minmax_scale = MinMaxScaler()
        X_minmax = minmax_scale.fit_transform(train_scale)
        X_minmax = pd.DataFrame(X_minmax)
        X_minmax['SK_ID_CURR'] = train_2['SK_ID_CURR']


        train_4 = X_minmax.loc[X_minmax['SK_ID_CURR'] == int(autre_clients)]
        train_4 = train_4.drop(columns=['SK_ID_CURR'])
        train_4 = pd.DataFrame(train_4.values).iloc[0]
        ###
        exp = lime1.explain_instance(train_4,
                                     model.predict_proba,
                                     num_samples=100)

        st.write('Hello, *World!* :sunglasses:')
        exp.show_in_notebook(show_table=True)
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        #exp.scvbbbave_to_file('lime_report.html')
        #st.image('./lime_report.html')
        # a completer l'id client pur le vrai
        fig.savefig('feature_importance_1'+'.png')
        st.image('./feature_importance_1.png')

#################################################