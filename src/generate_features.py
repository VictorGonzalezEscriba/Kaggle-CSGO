from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd


# DATA CLEANING
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
