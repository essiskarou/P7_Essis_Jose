import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import dill
import matplotlib.pyplot as plt
import lime
from lime import lime_tabular
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
import seaborn as sns
import joblib
import shap
import configparser
from hashlib import sha256

warnings.simplefilter(action='ignore', category=FutureWarning)

SEUIL = 0.5752224859896156

KEY = "P7"
PATH_PICKLE = 'pickle/'

app = Flask(__name__)

# chargemet du pickle du modèle
model = pickle.load(open(PATH_PICKLE+'model.pkl', 'rb'))
# chargemet de la base X_train
X_train = pickle.load(open(PATH_PICKLE+'X_train.pickle', 'rb'))
# chargemet de la base X
X = pickle.load(open(PATH_PICKLE+'X.pickle', 'rb'))

id_X = X.reset_index()
id_client = id_X['SK_ID_CURR']
df_id_client = pd.DataFrame(id_client)

# chargement des id clients
# on a limité volontairement la liste
@app.route('/api/clients')
def clients():
    client=  X.loc[100002.0:100012.0,:]
    #client_index = X.iloc[0:10,:]


    clients = list(clients.index.values)
    #client_index = list(client_index.index.values)
    # unique()
    #return jsonify(client)
    return clients


#@app.route('/api/indicator')
#def indicator():
#    id = request.args.get('id')
#
#    indic = request.args.get('indic')
#
#    id_client = int(id)
#    index = list(df_id_client[df_id_client.SK_ID_CURR == id_client].index)[0]
#    print("index:" + str(index))
#
#    print("indicator")
#
#    fig2, ax2 = plt.subplots()
#
#    x = X[[indic]]  # df_train_target1[col]
#    # x2 = df_train_target0[col]
#    # ax = sns.distplot(x, color = '#005800')
#    ax = sns.distplot(x, color='darkgreen')
#    # ax2 = sns.distplot(x2)
#    x = ax.lines[0].get_xdata()
#    y = ax.lines[0].get_ydata()
#    # plt.axvline(x[np.argmax(y)], color='red')
#    plt.axvline(float(X.iloc[index:index + 1][indic]), color='red')
#
#    ax.lines[0].remove()
    # ax2.lines[0].remove()
#    plt.show()
#    id_ = id + indic + KEY
#    sha = sha256(id_.encode('utf-8')).hexdigest()
#    fig2.savefig(sha + '.png')
#
#    return sha


# info du client  ( score, proba, etc )
@app.route('/api/client/<id_client>')
def client(id_client):
    print("id_client:<" + id_client + ">")
    # print(list(df_id_client.SK_ID_CURR))

    id_client = int(id_client)
    index = list(df_id_client[df_id_client.SK_ID_CURR == id_client].index)[0]
    print("index:" + str(index))
    y_pred = model.predict(X_train[index:index + 1, :])
    y_proba = model.predict_proba(X_train[index:index + 1, :])

    dico = {}
    dico["score"] = int((y_proba[:, 1] >= SEUIL).astype(int))  # str(y_pred[0])
    dico["proba0"] = str(round(y_proba[0][0] * 100, 2))
    dico["proba1"] = str(round(y_proba[0][1] * 100, 2))
    # dico["seuil"] = str(100 - SEUIL*100)
    dico["seuil"] = str(SEUIL * 100)

    dico["json"] = (X.iloc[index:index + 1]).to_json()

    #
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, X.columns)), columns=['Value', 'Feature'])
    #fig, ax = plt.subplots()

    #ax = sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[:20],
                     #color='#007500')
    #plt.title('Variables importantes')
    #plt.tight_layout()
    # plt.show()

    df_variables_princi = feature_imp.sort_values(by="Value", ascending=False)[:20].Feature

    #fig.savefig('feature_importance.png')

    # interpretablité du client en fct du  lime
    with open(PATH_PICKLE + 'lime_.pickle', 'rb') as f: lime1 = dill.load(f)

    exp = lime1.explain_instance(pd.DataFrame(X_train).iloc[index],
                                 model.predict_proba,
                                 num_samples=100)

    exp.show_in_notebook(show_table=True)
    #fig = exp.as_pyplot_figure()
    #plt.tight_layout()
    #fig.savefig('feature_importance_' + str(id_client) + '.png')

    df_variables = pd.DataFrame(exp.as_list())

    # liste des varaibles principales et liste des variables locales
    dico["variables"] = list(df_variables[0])
    dico["variables2"] = df_variables.to_json()
    dico["variables_princ"] = list(df_variables_princi)

    #plt.clf()

    #chargemet du pickle SHAP
    #explainer = shap.TreeExplainer(model)
    #Calculate Shap values
    #shap_values = explainer.shap_values( X.loc[100002.0:100012.0,:])
    #shap.initjs()

    #shap.summary_plot(shap_values,  X.loc[100002.0:100012.0,:], show=False)
    #plt.savefig("modèle_global.jpg")

    # list(df_variables[0])

    return jsonify(dico)


if __name__ == '__main__':
    app.run(debug=True)



#info du client  ( score, proba, etc )
#pp.route('/api/client/<id_client>')
#f client(id_client):
#  print("id_client:<" + id_client + ">")
#  # print(list(df_id_client.SK_ID_CURR))

#  id_client = float([for id in clients_id])
#  index_client = float([for index in client_index])

#  #index = list(df_id_client[df_id_client.SK_ID_CURR == id_client].index)[0]
#  print("index client:" + str(index_client))
#  y_pred = modele.predict(X_train[index_client:index_client + 1, :])
#  y_proba = modele.predict_proba(X_train[index_client:index_client + 1, :])

#pp.route('/predict', methods=['POST'])
#f predict():
#  '''
#  For rendering results on HTML GUI
#  '''
#  float_features = [float(x) for x in request.form.values()]
#  final_features = [np.array(float_features)]
#  prediction = model.predict(final_features)

#  if prediction == 0:
#      prediction = "solvable"
#      # pr = prob[0][0]
#  else:
#      prediction = "non solvable"
#      # pr = prob[0][0]

#  # output = round(prediction[0], 2)

#  return render_template('index.html', prediction_text='Cette personne est {}'.format(prediction))

#pp.route('/predict_api', methods=['POST'])
#f predict_api():
#  '''
#  For direct API calls trought request
#  '''
#  data = request.get_json(force=True)
#  prediction = model.predict([np.array(list(data.values()))])

#  # output = prediction[0]
#  return jsonify(prediction)

#  # plt.show()


#chargemet du pickle LIME_
#p = lime1.explain_instance(pd.DataFrame(X_train).iloc[0],
#                       model.predict_proba,
#                       num_samples=100)
#p.show_in_notebook(show_table=True)
#ig, axs = plt.subplots(2)
#g = exp.as_pyplot_figure()
#lt.tight_layout()
#g.savefig('modèle_local_'+str(0)+'.png')
#t.clf()

#chargemet du pickle SHAP
#plainer = shap.TreeExplainer(model)
#Calculate Shap values
#ap_values = explainer.shap_values(X.loc[[100002.0]])
#ap.initjs()

#ap.summary_plot(shap_values, X, show=False)
#t.savefig("modèle_global.jpg")

#pp.route('/')
#f home():
#  return render_template('index.html')

#pp.route('/predict',methods=['POST'])
#f predict():
#  '''
#  For rendering results on HTML GUI
#  '''
#  float_features = [float(x) for x in request.form.values()]
#  final_features = [np.array(float_features)]
#  prediction = model.predict(final_features)

#  if  prediction == 0:
#      prediction = "solvable"
#      # pr = prob[0][0]
#  else:
#      prediction = "non solvable"
#      # pr = prob[0][0]

#  #output = round(prediction[0], 2)

#  return render_template('index.html', prediction_text='Cette personne est {}'.format(prediction))

#pp.route('/predict_api',methods=['POST'])
#f predict_api():
#  '''
#  For direct API calls trought request
#  '''
#  data = request.get_json(force=True)
#  prediction = model.predict([np.array(list(data.values()))])

#  #output = prediction[0]
#  return jsonify(prediction)


# __name__ == "__main__":
# app.run(debug=True)