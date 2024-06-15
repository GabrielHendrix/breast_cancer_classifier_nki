# Módulo de Treinamento e Avaliação de Dados NKI

**Este módulo fornece uma estrutura para carregar, treinar e avaliar modelos de aprendizado de máquina no conjunto de dados NKI usando TensorFlow/Keras. Inclui funcionalidades para pré-processamento de dados, treinamento de modelos com agendamento de taxa de aprendizado e parada antecipada, e avaliação com várias métricas.**

## Índice

- **Instalação**
- **Uso**
- **Scripts**
    - **data.py**
    - **train.py**
    - **evaluate.py**
    - **main.py**
- **Parâmetros**
- **Resultados**

## Instalação

1. Clone o repositório:

```
    git clone https://github.com/seunomeusuario/nki-data-training.git
    cd nki-data-training
```

2. Instale os pacotes necessários:

```
    pip install -r requirements.txt
```


## Uso

O script principal main.py orquestra todo o processo de carregamento de dados, treinamento do modelo e avaliação dos resultados. Você pode executá-lo a partir da linha de comando e passar vários parâmetros para controlar seu comportamento.
Comando de Exemplo

```
    python main.py --kfold 5 --init_lr 0.001 --end_lr 0.0001 --lr_decay 2 --step_size 100 --epochs 5000
```


## Scripts

### data.py

Este script lida com o carregamento, pré-processamento e divisão dos dados em conjuntos de treinamento e teste. Também pode realizar a divisão para validação cruzada k-fold.

- #### Data:
    - **split_data()**: Divide os dados em conjuntos de treinamento e teste.
    - **load_dataset()**: Carrega e pré-processa o conjunto de dados NKI.
    - **split_kfold_data(n)**: Divide os dados para validação cruzada k-fold.
    - **load_splitted_datasets(i)**: Carrega conjuntos de dados previamente divididos do disco.

### train.py

Este script define a arquitetura do modelo e lida com o processo de treinamento, incluindo agendamento da taxa de aprendizado e parada antecipada.


- #### Train:
    - **__init__()**: Inicializa os parâmetros de treinamento.
    - **create_model()**: Define a arquitetura do modelo.
    - **learning_rate_scheduler()**: Define o agendamento da taxa de aprendizado.
    - **train()**: Treina o modelo e retorna o modelo treinado e o histórico.

### evaluate.py

Este script fornece funções para avaliar o modelo treinado usando várias métricas e gráficos.

- #### Evaluate:
    - **plot_metrics(histories, output_path)**: Plota perda e acurácia de treinamento e validação.
    - **model_evaluate(model, X_train, X_test, y_train, y_test, fold_no, output_path)**: Avalia o modelo e salva métricas no disco.

### main.py

O script principal que integra tudo. Ele lida com o parsing de argumentos, carregamento de dados, treinamento e avaliação.

    
- **main()**: Ponto de entrada principal do script. Faz o parsing dos argumentos e orquestra o fluxo de trabalho.

## Parâmetros

Os seguintes parâmetros podem ser passados para main.py:

- **--kfold**: Número de folds para validação cruzada (default: 1).
- **--input_path**: Caminho para o diretório contendo dados de treinamento salvos.
- **--init_lr**: Taxa de aprendizado inicial (default: 0.001).
- **--end_lr**: Taxa de aprendizado final para decaimento (default: 0.0001).
- **--lr_decay**: Fator de decaimento da taxa de aprendizado (default: 2).
- **--step_size**: Número de épocas antes de aplicar o decaimento da taxa de aprendizado (default: 100).
- **--epochs**: Número máximo de épocas para treinamento (default: 5000).

## Resultados

Após a execução do processo de treinamento e avaliação, os resultados, incluindo modelos treinados e métricas de avaliação, serão salvos no diretório de saída especificado. Gráficos de perda e acurácia ao longo das épocas também serão salvos.

