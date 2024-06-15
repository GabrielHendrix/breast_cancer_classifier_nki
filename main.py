from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from yellowbrick.features.rankd import Rank2D 
from yellowbrick.features.radviz import RadViz 
from yellowbrick.features.pcoords import ParallelCoordinates 
from time import localtime, strftime
import matplotlib.pyplot as plt 
import tensorflow as tf
import numpy as np
import pickle
import argparse
import data
import os


# Plotar as métricas de perda e acurácia para cada fold
def plot_metrics(histories, output_path):
    plt.figure(figsize=(14, 5))
    
    # Plotar a perda
    plt.subplot(1, 2, 1)
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=f'Fold {i+1} Treino')
        plt.plot(history.history['val_loss'], label=f'Fold {i+1} Validação', linestyle='dashed')
    plt.title('Perda durante o treinamento e validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    
    # Plotar a acurácia
    plt.subplot(1, 2, 2)
    for i, history in enumerate(histories):
        plt.plot(history.history['accuracy'], label=f'Fold {i+1} Treino')
        plt.plot(history.history['val_accuracy'], label=f'Fold {i+1} Validação', linestyle='dashed')
    plt.title('Acurácia durante o treinamento e validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'Results.png'))
    
    plt.show()


# Definir a arquitetura do modelo MLP com parâmetros para learning rate
def create_model(learning_rate, X):
    model = Sequential()
    model.add(Dense(30, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Callback para ajustar a learning rate
def learning_rate_scheduler(epoch, lr, step_size, end_lr, lr_decay):
    if epoch % step_size == 0 and epoch:
        new_lr = lr / lr_decay
        if new_lr < end_lr:
            new_lr = end_lr
        return (new_lr);
    return (lr);


def model_evaluate(model, X_train, X_test, y_train, y_test, fold_no, output_path):
    # visualise class separation
    classes = ['alive', 'dead']
    features = ['age', 'survival', 'timerecurrence', 'chemo', 'hormonal', 'amputation',
       'histtype', 'diam', 'posnodes', 'grade', 'angioinv', 'lymphinfil',
       'barcode']
    visualizer = RadViz(clases=classes, features=features)

    X_matrix = X_train.values
    y_matrix = np.array([v[0] for v in y_train.values])
    
    visualizer.fit(X_matrix, y_matrix)
    visualizer.transform(X_matrix)
    # visualizer.poof()
    visualizer.fig.savefig(os.path.join(output_path, 'train_data.png')) 

    # Supõe que y_test são os rótulos reais e y_pred são as previsões do modelo
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)

    # Acurácia
    accuracy = accuracy_score(y_test, y_pred_classes)
    print(f'Acurácia: {accuracy}')

    # Matriz de Confusão
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    print('Matriz de Confusão:')
    print(conf_matrix)

    # Relatório de Classificação
    class_report = classification_report(y_test, y_pred_classes)
    print('Relatório de Classificação:')
    print(class_report)

    # AUC-ROC
    auc = roc_auc_score(y_test, y_pred)
    print(f'AUC-ROC: {auc}')

    # Avaliação do modelo
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f'Fold {fold_no} - Acurácia: {scores[1]}')
    fold_no += 1


# Funcao de treino que salva o historico da loss e acuracia
def train(X, X_train, X_test, y_train, y_test):
    initial_learning_rate = 0.001
    # Parâmetros da learning rate
    lr_decay = 2.0
    end_learning_rate = 0.0001
    step_size = 100
    model = create_model(initial_learning_rate, X)

    # Callback para ajustar a learning rate
    lr_callback = LearningRateScheduler(lambda epoch, lr: learning_rate_scheduler(epoch, lr, step_size, end_learning_rate, lr_decay))

    # Callback de EarlyStopping
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Treinamento do modelo
    history = model.fit(X_train, y_train, epochs=5000, batch_size=128, verbose=1, validation_data=(X_test, y_test), callbacks=[lr_callback, early_stopping_callback])

    return model, history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',       type=bool,  default=False,  help='')
    parser.add_argument('--kfold',      type=int,   default=1,      help='Numero de folds para a validação cruzada')
    parser.add_argument('--input_path', type=str,   default=None,   help='Caminho para o diretorio que contem os dados salvos do treino a ser analisado')  # file/folder, 0 for webcam

    opt = parser.parse_args()
    output_path = os.path.join('checkpoints', strftime("%y%m%d_%H%M%S", localtime()))
    d = data.Data(output_path)
    histories = []
    
    if (opt.input_path is not None):
        output_path = opt.input_path
        input_path = opt.input_path
    elif (opt.input_path is None):
        input_path = output_path
        d.load_dataset()
        if (opt.kfold > 1):
            print('kfold')
            d.split_kfold_data(opt.kfold)
        else:
            d.split_data()

    d.load_splitted_datasets(input_path)  
    print(input_path)  
    if os.path.exists(input_path + '/folds'):
        input_path += '/folds'
        with os.scandir(input_path) as folds:
            for i, fold in enumerate(folds):
                print(i)
                model, history = train(d.X, d.X_train[i], d.X_test[i], d.y_train[i], d.y_test[i])
                histories.append(history)
                with open(os.path.join(input_path, fold.name, 'trainHistoryDict'), 'wb') as file_pi:
                    pickle.dump(histories, file_pi)
                model.save(os.path.join(input_path, fold.name, 'model.h5'))
                model_evaluate(model, d.X_train[i], d.X_test[i], d.y_train[i], d.y_test[i], i, input_path)

    elif os.path.exists(input_path):
        i = 0
        model, history = train(d.X, d.X_train[i], d.X_test[i], d.y_train[i], d.y_test[i])
        histories.append(history)
        with open(os.path.join(input_path, 'trainHistoryDict'), 'wb') as file_pi:
                pickle.dump(histories, file_pi)
        model.save(os.path.join(input_path, 'model.h5'))
        model_evaluate(model, d.X_train[i], d.X_test[i], d.y_train[i], d.y_test[i], i, input_path)

    plot_metrics(histories, output_path)


if __name__ == "__main__":
    main()









