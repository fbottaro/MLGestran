
# coding: utf-8

# In[1]:


#LEIA OS COMENTÁRIOS

# importando das bibliotecas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

#leitura do csv, data para análise
balance_data = pd.read_csv('C:/Users/Fernando/Downloads/result2.csv', sep= ';')

print "Dataset Lenght:: ", len(balance_data)
print "Dataset Shape:: ", balance_data.shape

# X são os valores de latilong [00.000000 -00.000000 ] 
# you can print if you want
# print X
X = balance_data.values[:,0:2]

# Y são os resultados esperados ['PR SC SP'] 
# you can print if you want
# print X
Y = balance_data.values[:,2]

# aqui é a separação dos dados para treinar o modelo e depois para testa-lo 
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

# classificador TREE com creterion "gini"
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

# classificador TREE com creterion "entropy"
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

# predição com base no array de testes, um para cada classificador
y_pred = clf_gini.predict(X_test)
y_pred_en = clf_entropy.predict(X_test)

#IMPORTANTE: olhar a documentação para entender melhor os parametros
#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

print "Accuracy is ", accuracy_score(y_test,y_pred)*100
print "Accuracy is ", accuracy_score(y_test,y_pred_en)*100


# In[2]:


# script para prever os proximos 3 ltlg com base no modelo treinado

# le os dados
#Curitiba Latitude:-25.441105 Longitude:-49.276855

lat = float(input("lat"))
lng = float(input("lng"))
#predição
result = clf_gini.predict([[lat, lng]])
#imprime o resultado
print result

#Campinas Latitude:-22.907104 Longitude:-47.063240

#same thing
lat = float(input("lat"))
lng = float(input("lng"))
result = clf_gini.predict([[lat, lng]])
print result

#Florianópolis Latitude:-27.593500 Longitude:-48.558540

#same thing
lat = float(input("lat"))
lng = float(input("lng"))
result = clf_gini.predict([[lat, lng]])
print result

# Obs: O modelo preve somente 3 estados, o modelo não foi treinado para prever por exemplo o ltlg de Fortaleza 
# Cheers - qualquer dúvida me chama no whats

