import pandas as pd
import numpy as np
import random
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

sns.set(rc={'figure.figsize':(11.7,8.27)})

def load_dataset(path):
    df = pd.read_csv(path, delimiter=',', engine='python')
    return df

dataset = load_dataset('./data/csgo_round_snapshots.csv')
# Veiem les dimensionalitats de la base de dades i les 5 primeres files, per veure el format.
print("Dimensionalitat de la BBDD: ", dataset.shape)
dataset.head()

# DATA UNDERSTANDING
# Per veure la distibució de qui guanya més partides
sns.histplot(data=dataset, x='round_winner')

# Per veure la distribució dels mapes
sns.histplot(data = dataset, x = 'map')

d1 = dataset[dataset['map'] == 'de_dust2']
d2 = dataset[dataset['map'] == 'de_inferno']
# dust2
sns.histplot(data=d1, x='round_winner')
# inferno
sns.histplot(data=d2, x='round_winner')

row = random.randint(0, dataset.shape[0])
sample = dataset.iloc[[row]]
ct_money, t_money = sample['ct_money'], sample['t_money']
print('ct_money: ', ct_money)
print('t_money: ', t_money)


# DATA CLEANING
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

# IMPLEMENTATION


X = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# LOGISTIC REGRESSION
Estimator1 = LogisticRegression()
param_grid1 = {'fit_intercept':[True, False]}

# Utilitzo aquest optimitzador, vist a la documentació de Kaggle
Optimizer1 = GridSearchCV(Estimator1, param_grid1, cv=5)
Optimizer1.fit(X_train, y_train)

predsTrain = Optimizer1.predict(X_train)
predsTest = Optimizer1.predict(X_test)
print('Linear Regression')
print(classification_report(y_train, predsTrain))
print(classification_report(y_test, predsTest))


# DECISION TREE
Estimator2 = DecisionTreeClassifier()
param_grid2 = {'max_depth':[None,1,2,3], 'min_samples_leaf' :[2,3,4]}

# Utilitzo aquest optimitzador, vist a la documentació de Kaggle
Optimizer2 = GridSearchCV(Estimator2, param_grid2, cv = 5)
Optimizer2.fit(X_train, y_train)

predsTrain = Optimizer2.predict(X_train)
predsTest = Optimizer2.predict(X_test)
print('Decision Tree')
print(classification_report(y_train, predsTrain))
print(classification_report(y_test, predsTest))
