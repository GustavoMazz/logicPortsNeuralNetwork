# Executado com Python v3.7
from tensorflow import keras
import tensorflow as tf
import numpy as np
import sympy as sp

sp.init_printing(use_latex="mathjax")

class InvalidPortException(Exception):
    """ Exception quando é informado uma porta que o o sistema desconhece """
    pass

class NeuralNetwork():
    """ 
        Rede Neural para classificação do resultado da tabela verdade de uma porta lógica
            suporta: AND, OR e XOR
    """
    x_data = [[0,0], [0,1], [1,0], [1,1]]
    ports_y_data = {
        'AND': [[0], [0], [0], [1]],
        'OR': [[0], [1], [1], [1]],
        'XOR': [[0], [1], [1], [1]]
    }

    def __init__(self, port="AND"):
        """ Inicializa as variáveis """
        if not port in ['AND', 'OR', 'XOR']: raise InvalidPortException()

        self.x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
        self.y_data = np.array(self.ports_y_data[port])

    def createModel(self):
        """ Método para criação do modelo do Keras """

        self.model = keras.Sequential()

        self.model.add(keras.layers.Dense(32, activation="sigmoid",input_shape=(2,)))
        self.model.add(keras.layers.Dense(1, activation="sigmoid", input_shape=(1,)))
        optimizer = keras.optimizers.SGD(learning_rate=0.1)

        self.model.compile(optimizer=optimizer, loss="mse", metrics=['accuracy', 'mse'])

    def train(self):        
        """
            Método de treinamento
                TensorBoard callback: grava as métricas a cada época
                EarlyStopping verifica se nas ultimas 'patience' execuções melhorou pelo menos o valor do delta
                    e.g.: precisa ter melhorado pelo menos 0.0001 no valor do loss nas 5 últimas épocas para continuar treinando
        """
        tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.0001)

        self.model.fit(self.x_data, self.y_data, batch_size=4, epochs=5000, callbacks=[es_callback, tb_callback])

    def test(self):
        """ Método de teste e validação """
        y_pred = self.model.predict(self.x_data)
        mse = tf.keras.metrics.mean_squared_error(
            self.y_data, y_pred
        )
        print(mse)

        
if __name__ == '__main__':
    """ Porta de entrada do algoritmo, instancia a Rede Neural e executa os métodos necessários """

    # Requisita a porta lógica
    port = input("Porta lógica: ")

    nn = NeuralNetwork(port=port)
    nn.createModel()
    nn.train()
    nn.test()
        