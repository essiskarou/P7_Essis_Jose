import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import dill

#########################
SEUIL = 0.5752224859896156
PATH_PICKLE = 'pickle/'

########################
app = Flask(__name__)
# chargemet du pickle du modèle
model = pickle.load(open(PATH_PICKLE+'model.pkl', 'rb'))
# chargemet de la base X_train
X_train = pickle.load(open(PATH_PICKLE+'X_train.pickle', 'rb'))
# chargemet de la base X
X = pickle.load(open(PATH_PICKLE+'X.pickle', 'rb'))

train_2 = pickle.load(open(PATH_PICKLE+'train_2.pickle', 'rb'))

# chargemet de la base des voisins
neighbors = pickle.load(open(PATH_PICKLE+'neighbors.pickle', 'rb'))


#id_X = X.reset_index()
id_client = train_2['SK_ID_CURR']
df_id_client = pd.DataFrame(id_client)

# chargement des id clients
# on a limité volontairement la liste
@app.route('/api/clients')
def clients():
    client=  train_2.iloc[:20]
    #client_index = X.iloc[0:10,:]

    clients = list(client['SK_ID_CURR'])

    clients = ','.join(str(l) for l in clients)

    return clients

# info du client  ( score, proba, etc )
@app.route('/api/client/<id_client>')
def client(id_client):
    print("id_client:<" + id_client + ">")

    id_client = int(id_client)
    index = list(df_id_client[df_id_client.SK_ID_CURR == id_client].index)[0]
    print("index:" + str(index))
    y_pred = model.predict(X_train[index:index + 1, :])
    y_proba = model.predict_proba(X_train[index:index + 1, :])

    dico = {}
    dico["score"] = int((y_proba[:, 1] >= SEUIL).astype(int))
    dico["proba0"] = str(round(y_proba[0][0] * 100, 2))
    dico["proba1"] = str(round(y_proba[0][1] * 100, 2))
    dico["seuil"] = str(SEUIL * 100)
    dico["json_1"] = neighbors.to_json()


    dico["json"] = (X.iloc[index:index + 1]).to_json()

    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, X.columns)), columns=['Value', 'Feature'])

    df_variables_princi = feature_imp.sort_values(by="Value", ascending=False)[:20].Feature

    dico["variables_princ"] = list(df_variables_princi)

    return jsonify(dico)

if __name__ == '__main__':
    app.run(debug=True)