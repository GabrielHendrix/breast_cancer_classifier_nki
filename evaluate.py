from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from yellowbrick.features.radviz import RadViz 
import matplotlib.pyplot as plt
import numpy as np
import os

class Evaluate:
    # Plotar as métricas de perda e acurácia para cada fold
    def plot_metrics(self, histories, output_path):
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


    def model_evaluate(self, model, X_train, X_test, y_train, y_test, fold_no, output_path):
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
        with open(os.path.join(output_path, 'evaluate_results.txt'), 'a') as file:
            # Acurácia
            accuracy = accuracy_score(y_test, y_pred_classes)
            print(f'Acurácia: {accuracy}')
            file.write('Acurácia: ' + str(accuracy) + '\n\n')

            # Matriz de Confusão
            conf_matrix = confusion_matrix(y_test, y_pred_classes)
            print('Matriz de Confusão:')
            print(conf_matrix)
            file.write('Matriz de Confusão:\n')
            file.write(str(conf_matrix) + '\n\n')

            # Relatório de Classificação
            class_report = classification_report(y_test, y_pred_classes)
            print('Relatório de Classificação:')
            print(class_report)
            file.write('Relatório de Classificação:\n')
            file.write(str(class_report) + '\n\n')

            # AUC-ROC
            auc = roc_auc_score(y_test, y_pred)
            print(f'AUC-ROC: {auc}')
            file.write('AUC-ROC: ' + str(auc) + '\n\n')

            # Avaliação do modelo
            scores = model.evaluate(X_test, y_test, verbose=0)
            if fold_no == -1:
                print(f'Acurácia: {scores[1]}')
                file.write('Acurácia: ' + str(scores[1]) + '\n')
            else:
                print(f'Fold {fold_no} - Acurácia: {scores[1]}')
                file.write('Fold ' + str(fold_no) + ' - Acurácia: ' + str(scores[1]) + '\n')