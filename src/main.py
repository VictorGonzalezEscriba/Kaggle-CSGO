import pandas as pd
import random
import seaborn as sns
import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from generate_features import *


sns.set(rc={'figure.figsize': (11.7, 8.27)})

def load_dataset(path):
    df = pd.read_csv(path, delimiter=',', engine='python')
    return df


dataset = load_dataset('./data/csgo_round_snapshots.csv')
# Veiem les dimensionalitats de la base de dades i les 5 primeres files, per veure el format.
print("Dimensionalitat de la BBDD: ", dataset.shape, "\n")
dataset.head()

# DATA UNDERSTANDING

# Per veure la distibució de qui guanya més partides
sns.histplot(data=dataset, x='round_winner').set_title('Quantitat de partides guanaydes per equip')
plt.show()

# Per veure la distribució dels mapes
sns.histplot(data=dataset, x='map').set_title('Distribució dels mapes')
plt.show()

d1 = dataset[dataset['map'] == 'de_dust2']
d2 = dataset[dataset['map'] == 'de_inferno']

# dust2
sns.histplot(data=d1, x='round_winner').set_title('Quantitat de partides guanyades al mapa dust2')
plt.show()

# inferno
sns.histplot(data=d2, x='round_winner').set_title('Quantitat de partides guanyades al mapa inferno')
plt.show()

# Per comparar els diners dels dos equips en una mostra aleatoria
row = random.randint(0, dataset.shape[0])
sample = dataset.iloc[[row]]
ct_money, t_money = sample['ct_money'], sample['t_money']
print('ct_money: ', ct_money)
print('t_money: ', t_money, "\n")

# IMPLEMENTATION
# clean the dataset
dataset, X, y = clean_dataset(dataset)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# LOGISTIC REGRESSION
start1 = time.time()
LR = LogisticRegression(max_iter=1000)
param_grid1 = {'fit_intercept': [True, False]}

# Utilitzo aquest optimitzador, vist a la documentació de Kaggle
Optimizer1 = GridSearchCV(LR, param_grid1, cv=5)
Optimizer1.fit(X_train, y_train)

predsTrain1 = Optimizer1.predict(X_train)
predsTest1 = Optimizer1.predict(X_test)
print('Linear Regression:')
print('Train:')
print(classification_report(y_train, predsTrain1))
print('Test:')
print(classification_report(y_test, predsTest1))
end1 = time.time()
print("Time LR: ", end1-start1)


# DECISION TREE
start2 = time.time()
DT = DecisionTreeClassifier()
param_grid2 = {'max_depth': [None, 1, 2, 3], 'min_samples_leaf': [2, 3, 4]}

# Utilitzo aquest optimitzador, vist a la documentació de Kaggle
Optimizer2 = GridSearchCV(DT, param_grid2, cv=5)
Optimizer2.fit(X_train, y_train)

predsTrain2 = Optimizer2.predict(X_train)
predsTest2 = Optimizer2.predict(X_test)
print('Decision Tree:')
print('Train:')
print(classification_report(y_train, predsTrain2))
print('Test:')
print(classification_report(y_test, predsTest2))
end2 = time.time()
print("Time DT: ", end2-start2)


# KNN
start3 = time.time()
acc, t_acc = [], 0
prec, t_prec = [], 0
rec, t_rec = [], 0
n = [5, 6, 7, 8]
for i in n:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    acc.append(accuracy_score(y_test, predictions))
    prec.append(precision_score(y_test, predictions, average='micro'))
    rec.append(recall_score(y_test, predictions, average='micro'))

for i in range(0, len(n)):
    t_acc += acc[i]
    t_prec += prec[i]
    t_rec += rec[i]

print('KNN:')
print('Mean Accuracy: ', t_acc / len(n))
print('Mean Precision: ', t_prec / len(n))
print('Mean Recall: ', t_rec / len(n))
end3 = time.time()
print("Time KNN: ", end3-start3)

