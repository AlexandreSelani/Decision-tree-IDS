from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import f1_score,accuracy_score
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

"""Código para treinar e testar uma arvore de decisao para detectar ataques de rede.

Baseado no dataset IoT intrusion (https://www.kaggle.com/datasets/subhajournal/iotintrusion)"""

def loadData(data):

    y = data.iloc[:,-1]

    X = data.iloc[:,:-1]
    
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

    return skf,X,y


def saveTreeImage(decision_tree):
    plt.figure(figsize=(50,50))
    tree.plot_tree(decision_tree, max_depth=3, filled=True, fontsize=10)
    plt.savefig("arvore_exemplo.png", dpi=300, bbox_inches="tight")

def featureImportances(decision_tree):

    feature_importances = decision_tree.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    print(importance_df)

if __name__ == "__main__":

    #-----------------Carregamento de dados e pré-processamento
    try:
        raw_data = pd.read_csv('IoT_Intrusion.csv')
    except Exception as e:
        print("Erro ao carregar arquivo")
        exit()


    skf,X,y = loadData(raw_data)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    #------------------Treinamento + teste e avaliação (usando Stratified k-fold cross validation)
    min_samples_split=100
    accs,f1s = np.array([]),np.array([])
    
    for i, (train_idx, test_idx) in enumerate(skf.split(X,y)):
        X_train = X.iloc[train_idx,:]
        y_train = y[train_idx]

        X_test = X.iloc[test_idx,:]
        y_test = y[test_idx]

        decision_tree = tree.DecisionTreeClassifier()
        decision_tree.fit(X_train,y_train)

        if(i==0):
            saveTreeImage(decision_tree)

        predicts = decision_tree.predict(X_test)

        F1_score = f1_score(y_test, predicts,average='macro')
        acc = accuracy_score(y_test, predicts)

        f1s = np.append(f1s,F1_score)
        accs = np.append(acc,acc)
        print(f"FOLD {i}\nAccuracy = {acc}\nF1 score = {F1_score}\n-------------------------")

    
        #----------------Importancia das features
        #featureImportances(decision_tree)

    print(f"Resultados finais:\nAccuracy = {accs.mean():.3f}\nF1 score = {f1s.mean():.3f}")


