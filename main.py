from time import localtime, strftime
import argparse
import evaluate
import pickle
import train
import data
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kfold',      type=int,   default=1,      help='Numero de folds para a validação cruzada')
    parser.add_argument('--input_path', type=str,   default=None,   help='Caminho para o diretorio que contem os dados salvos do treino a ser analisado')
    parser.add_argument('--init_lr',    type=float, default=0.001,  help='Valor da taxa de aprendizado inicial')
    parser.add_argument('--end_lr',     type=float, default=0.0001, help='Valor da taxa de aprendizado inicial')
    parser.add_argument('--lr_decay',   type=float, default=2.000,  help='Taxa de decaimento (learning_rate / learning_rate_decay)')
    parser.add_argument('--step_size',  type=int,   default=100,    help='Define o numero de epocas que devem ser acumuladas para aplicar um decaimento sobre o learning_rate')
    parser.add_argument('--epochs',     type=int,   default=5000,   help='Numero maximo de epocas')
    parser.add_argument('--bad_train',  type=bool,  default=False,   help='Habilita a formacao de um conjuntos de dados ruim para testes')
    opt = parser.parse_args()

    output_path = os.path.join('checkpoints', strftime("%y%m%d_%H%M%S", localtime()))
    d = data.Data(output_path)
    
    if (opt.input_path is not None):
        output_path = opt.input_path
        input_path = opt.input_path
    
    elif (opt.input_path is None):
        input_path = output_path
        d.load_dataset()
        
        if (opt.kfold > 1):
            d.split_kfold_data(opt.kfold)
        else:
            d.split_data(opt.bad_train)

    d.load_splitted_datasets(input_path)  
    e = evaluate.Evaluate()
    histories = []
    
    if os.path.exists(input_path + '/folds'):
        input_path += '/folds'
        with os.scandir(input_path) as folds:
            for i, fold in enumerate(folds):
                tr = train.Train(d.X, d.X_train[i], d.X_test[i], d.y_train[i], d.y_test[i], opt.init_lr, opt.lr_decay, opt.end_lr, opt.step_size, opt.epochs)
                model, history = tr.train()
                histories.append(history)
                with open(os.path.join(input_path, fold.name, 'trainHistoryDict'), 'wb') as file_pi:
                    pickle.dump(histories, file_pi)
                model.save(os.path.join(input_path, fold.name, 'model.h5'))
                e.model_evaluate(model, d.X_train[i], d.X_test[i], d.y_train[i], d.y_test[i], i, input_path)

    elif os.path.exists(input_path):
        i = 0
        tr = train.Train(d.X, d.X_train[i], d.X_test[i], d.y_train[i], d.y_test[i], opt.init_lr, opt.lr_decay, opt.end_lr, opt.step_size, opt.epochs)
        model, history = tr.train()
        histories.append(history)
        
        # Salva o historico da loss e da acuracia obtidos durante o treino
        with open(os.path.join(input_path, 'trainHistoryDict'), 'wb') as file_pi:
                pickle.dump(histories, file_pi)
        model.save(os.path.join(input_path, 'model.h5'))
        
        # Calcula a matriz de confusao e as metricas precision, recall, f1-score e AUC-ROC
        e.model_evaluate(model, d.X_train[i], d.X_test[i], d.y_train[i], d.y_test[i], -1, input_path)

    e.plot_metrics(histories, output_path)


if __name__ == "__main__":
    main()









