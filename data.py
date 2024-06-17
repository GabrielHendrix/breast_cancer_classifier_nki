from sklearn.model_selection import train_test_split, StratifiedKFold
from yellowbrick.features.pcoords import ParallelCoordinates 
from sklearn.preprocessing import StandardScaler
from yellowbrick.features.radviz import RadViz 
from yellowbrick.features.rankd import Rank2D 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class Data:
    def __init__(self, out_path):
        self.output_path    = out_path
        self.X              = None
        self.y              = None
        self.X_train        = []
        self.X_test         = []
        self.y_train        = []
        self.y_test         = []


    def load_splitted_datasets(self, i):
        input_path = i
        self.X = pd.read_parquet(os.path.join(input_path, 'X.parquet'))
        if os.path.exists(input_path + '/folds'):
            input_path += '/folds'
   
            # with os.scandir(input_path) as folds:
            for fold in sorted(os.listdir(input_path)):
                print(os.path.join(input_path, fold))
                with os.scandir(os.path.join(input_path, fold)) as files:
                    for file in files:
                        if file.name == 'X_train.parquet':
                            self.X_train.append(pd.read_parquet(os.path.join(input_path, fold, file.name)))
                        if file.name == 'X_test.parquet':
                            self.X_test.append(pd.read_parquet(os.path.join(input_path, fold, file.name)))
                        if file.name == 'y_train.parquet':
                            self.y_train.append(pd.read_parquet(os.path.join(input_path, fold, file.name)))
                        if file.name == 'y_test.parquet':
                            self.y_test.append(pd.read_parquet(os.path.join(input_path, fold, file.name)))
        else:
            self.X_train.append(pd.read_parquet(os.path.join(input_path, 'X_train.parquet')))
            self.X_test.append(pd.read_parquet(os.path.join(input_path, 'X_test.parquet')))
            self.y_train.append(pd.read_parquet(os.path.join(input_path, 'y_train.parquet')))
            self.y_test.append(pd.read_parquet(os.path.join(input_path, 'y_test.parquet')))


    def load_dataset(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        pd.set_option('display.max_columns', None)

        # Carregar a base de dados
        data_raw = pd.read_csv("NKI_cleaned.csv")

        # Seleciona os atributos
        features_to_drop    = data_raw.columns[16:]
        data_subset         = data_raw.drop(features_to_drop, axis=1)

        # Define a coluna que sera o label
        self.X = data_subset.drop(['Patient', 'ID', 'eventdeath'], axis=1)
        self.y = data_subset['eventdeath']
        
        self.X.to_parquet(os.path.join(self.output_path, 'X.parquet'))

        # Normalizar os dados
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)


    def split_kfold_data(self, n):
        # Configurações da validação cruzada
        kfold       = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)
        fold_no     = 1

        folds_path = os.path.join(self.output_path, 'folds')
        if not os.path.exists(folds_path):
            os.makedirs(folds_path)

        for train, test in kfold.split(self.X, self.y):
            out_path = os.path.join(folds_path, str(fold_no))
            os.makedirs(out_path)
            # print(out_path)
            pd.DataFrame(self.X[train]).to_parquet(os.path.join(out_path, 'X_train.parquet'))
            pd.DataFrame(self.X[test]).to_parquet(os.path.join(out_path, 'X_test.parquet'))
            pd.DataFrame(self.y[train]).to_parquet(os.path.join(out_path, 'y_train.parquet'))
            pd.DataFrame(self.y[test]).to_parquet(os.path.join(out_path, 'y_test.parquet'))
            fold_no += 1
        

    def split_data(self, bad_train):
        if bad_train:
            # Aplicar KMeans para agrupar os dados
            kmeans = KMeans(n_clusters=13, random_state=42)
            clusters = kmeans.fit_predict(self.X)
            distances = kmeans.transform(self.X)
            # Ordenar os dados com base na distância ao centro do cluster
            sorted_indices = np.argsort(np.min(distances, axis=1))
            # Selecionar os dados mais próximos ao centro dos clusters para treino
            train_size = int(0.8 * len(self.X))
            train_indices = sorted_indices[:train_size]
            test_indices = sorted_indices[train_size:]

            X_train = self.X[train_indices]
            X_test = self.X[test_indices]
            y_train = self.y.iloc[train_indices]
            y_test = self.y.iloc[test_indices]
        else:
            # Divide entre treino e teste
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
       


        pd.DataFrame(X_train).to_parquet(os.path.join(self.output_path, 'X_train.parquet'))
        pd.DataFrame(X_test).to_parquet(os.path.join(self.output_path, 'X_test.parquet'))
        pd.DataFrame(y_train).to_parquet(os.path.join(self.output_path, 'y_train.parquet'))
        pd.DataFrame(y_test).to_parquet(os.path.join(self.output_path, 'y_test.parquet'))