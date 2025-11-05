from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
"""Código para treinar e testar uma arvore de decisao para detectar ataques de rede.

Baseado no dataset IoT intrusion (https://www.kaggle.com/datasets/subhajournal/iotintrusion)"""

def loadData(data):

    labels = data.iloc[:,-1]

    loaded_data = data.iloc[:,:-1]
    print(labels.unique())
    X_train,X_test,y_train,y_test = train_test_split(loaded_data,labels,test_size=0.20)
    return X_train,X_test,y_train,y_test

def saveTreeImage(decision_tree):
    plt.figure(figsize=(50,50))
    tree.plot_tree(decision_tree, max_depth=10, filled=True, fontsize=8)
    plt.savefig("arvore_reduzida.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    #-----------------Carregamento de dados e pré-processamento
    try:
        raw_data = pd.read_csv('IoT_Intrusion.csv')
    except Exception as e:
        print("Erro ao carregar arquivo")
        exit()


    X_train,X_test,y_train,y_test = loadData(raw_data)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)
    

    #------------------Treinamento + teste e avaliação
    min_samples_split=100
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree.fit(X_train,y_train)

    saveTreeImage(decision_tree)

    predicts = decision_tree.predict(X_test)
    F1_score = f1_score(y_test, predicts,average='macro')
    acc = accuracy_score(y_test, predicts)

    print(f"accuracy:{acc}\nF1 score = {F1_score}")

    #----------------Importancia das features

    feature_importances = decision_tree.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print(importance_df)




