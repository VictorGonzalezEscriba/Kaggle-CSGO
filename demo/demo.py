import pandas as pd
import random
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

sns.set(rc={'figure.figsize': (11.7, 8.27)})

def clean_dataset(dataset):
    encoder = LabelEncoder()
    scaler = StandardScaler()

    # Eliminació de columnes amb un únic valor
    for column in dataset.columns:
        if len(dataset[column].unique()) == 1:
            dataset = dataset.drop([column], axis=1)

    # Sustituio l'equip guanyador per 0 (T) i 1 (CT)
    # Passo la columna bomb_planted de tipus booleà a tipus int
    dataset['round_winner'] = dataset['round_winner'].replace({'T': 0, 'CT': 1})
    dataset['bomb_planted'] = dataset['bomb_planted'].astype(np.int16)
    y = dataset['round_winner']

    # Eliminem la columna objectiu per reduïr el dataset
    dataset = dataset.drop('round_winner', axis=1)
    dataset['map'] = encoder.fit_transform(dataset['map'])

    x = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)

    return dataset, x, y


def load_dataset(path):
    df = pd.read_csv(path, delimiter=',', engine='python')
    return df


dataset = load_dataset('./data/csgo_round_snapshots.csv')
dataset = dataset[:1000]
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
LR = LogisticRegression(max_iter=1000)
param_grid1 = {'fit_intercept': [True, False]}

# Utilitzo aquest optimitzador, vist a la documentació de Kaggle
Optimizer1 = GridSearchCV(LR, param_grid1, cv=5)
Optimizer1.fit(X_train, y_train)

predsTrain = Optimizer1.predict(X_train)
predsTest = Optimizer1.predict(X_test)
print('Linear Regression:')
print('Train:')
print(classification_report(y_train, predsTrain))
print('Test:')
print(classification_report(y_test, predsTest))


# DECISION TREE
DT = DecisionTreeClassifier()
param_grid2 = {'max_depth': [None, 1, 2, 3], 'min_samples_leaf': [2, 3, 4]}

# Utilitzo aquest optimitzador, vist a la documentació de Kaggle
Optimizer2 = GridSearchCV(DT, param_grid2, cv=5)
Optimizer2.fit(X_train, y_train)

predsTrain = Optimizer2.predict(X_train)
predsTest = Optimizer2.predict(X_test)
print('Decision Tree:')
print('Train:')
print(classification_report(y_train, predsTrain))
print('Test:')
print(classification_report(y_test, predsTest))


# KNN
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

