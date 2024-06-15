from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf

class Train:
    def __init__(self, X, X_train, X_test, y_train, y_test, init_lr, lr_decay, end_lr, step_size, epochs):
        self.X          = X
        self.X_train    = X_train
        self.X_test     = X_test
        self.y_train    = y_train
        self.y_test     = y_test

        # Parâmetros da learning rate
        self.init_lr    = init_lr   # 0.001
        self.lr_decay   = lr_decay  # 2.0
        self.end_lr     = end_lr    # 0.0001
        self.step_size  = step_size # 100
        self.epochs     = epochs    # 5000
   

    # Definir a arquitetura do modelo MLP com parâmetros para learning rate
    def create_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.X.shape[1], activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.init_lr)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model


    # Callback para ajustar a learning rate
    def learning_rate_scheduler(self, epoch, lr):
        if epoch % self.step_size == 0 and epoch:
            new_lr = lr / self.lr_decay
            if new_lr < self.end_lr:
                new_lr = self.end_lr
            return (new_lr);
        return (lr);


    # Funcao de treino que salva o historico da loss e acuracia
    def train(self):
        model = self.create_model()

        # Callback para ajustar a learning rate
        lr_callback = LearningRateScheduler(lambda epoch, lr: self.learning_rate_scheduler(epoch, lr))

        # Callback de EarlyStopping
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Treinamento do modelo
        history = model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=128, verbose=1, validation_data=(self.X_test, self.y_test), callbacks=[lr_callback, early_stopping_callback])

        return model, history
