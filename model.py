#!/usr/bin/env python3
import ucimlrepo as uci

from enum import Enum
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

EPOCHS = 100000
DEFAULT_LR = 1e-3

'''
A comparison of different machine learning optimization
algorithms using a toy linear regression model.
'''
def main():
    iris = uci.fetch_ucirepo(id=53)
    X = iris.data.features
    y = np.array(X['petal length'])
    X = np.array(X.drop(['petal length'], axis=1))

    linear = LinearRegression(feature_count=X.shape[1])

    adam_learning_curve = linear.train(X, y, optimizer=Optimizer.Adam, epochs=EPOCHS, lr=0.001)
    gd_learning_curve = linear.train(X, y, optimizer=Optimizer.GradientDescent, epochs=EPOCHS, lr=0.001)
    sgd_learning_curve = linear.train(X, y, optimizer=Optimizer.StochasticGradientDescent, epochs=EPOCHS, lr=0.001)

    fig = go.Figure()
    fig.add_scatter(x=list(range(EPOCHS)), y=adam_learning_curve, name='Adam', mode='lines')
    fig.add_scatter(x=list(range(EPOCHS)), y=gd_learning_curve, name='Gradient Descent', mode='lines')
    fig.add_scatter(x=list(range(EPOCHS)), y=sgd_learning_curve, name='Stochastic Gradient Descent', mode='lines')

    fig.update_xaxes(type='log')
    fig.update_layout(
        title='Linear Regression Optimization Curves',
        xaxis_title='Iterations',
        yaxis_title='MSE',
        font={
            'family': 'Hack, Monospace',
            'size': 18,
            'color': 'RebeccaPurple',
        }
    )

    fig.show()
    fig.write_image('linear-comparison.png', width=1920, height=1080)

    X = iris.data.features
    y = iris.data.targets
    X['class'] = y['class']

    X = X[X['class'] != 'Iris-virginica']
    y = np.array(X['class'].apply(lambda x: 1 if x == 'Iris-setosa' else 0))
    X = np.array(X.drop(['class'], axis=1))

    logistic = LogisticRegression(feature_count=X.shape[1])

    adam_learning_curve = logistic.train(X, y, optimizer=Optimizer.Adam, epochs=EPOCHS, lr=0.001)
    gd_learning_curve = logistic.train(X, y, optimizer=Optimizer.GradientDescent, epochs=EPOCHS, lr=0.001)
    sgd_learning_curve = logistic.train(X, y, optimizer=Optimizer.StochasticGradientDescent, epochs=EPOCHS, lr=0.001)

    fig = go.Figure()
    fig.add_scatter(x=list(range(EPOCHS)), y=adam_learning_curve, name='Adam', mode='lines')
    fig.add_scatter(x=list(range(EPOCHS)), y=gd_learning_curve, name='Gradient Descent', mode='lines')
    fig.add_scatter(x=list(range(EPOCHS)), y=sgd_learning_curve, name='Stochastic Gradient Descent', mode='lines')

    fig.update_xaxes(type='log')
    fig.update_layout(
        title='Logistic Regression Optimization Curves',
        xaxis_title='Iterations',
        yaxis_title='Log Loss',
        font={
            'family': 'Hack, Monospace',
            'size': 18,
            'color': 'RebeccaPurple',
        }
    )

    fig.show()
    fig.write_image('logistic-comparison.png', width=1920, height=1080)


def visualize_linear_regression(lg, x, y):
    predict_y = lg.predict(x)
    predict_df = pd.DataFrame.from_dict({ 'x': x, 'prediction': predict_y })
    actual_df = pd.DataFrame.from_dict({ 'x': x, 'y': y })

    fig = px.line(predict_df, x='x', y='prediction', title='Linear Regression')
    fig.add_scatter(x=actual_df['x'], y=actual_df['y'], mode='lines')
    fig.show()


def visualize_training_path(lg):
    weight_history = lg.past_weights

    data = pd.DataFrame.from_dict({
        'w': weight_history,
        'epoch': list(range(0, len(weight_history)))
    })
    fig = px.line(data, x='epoch', y='w')
    fig.show()


def generate_xor_data():
    # random sequence of ones and zeros...
    x_1 = np.random.randint(low=0, high=2, size=(100))
    x_2 = np.random.randint(low=0, high=2, size=(100))
    y = x_1 ^ x_2
    x = np.array([x_1, x_2]).T

    return x, y


def generate_noisy_data():
    x = np.arange(start=0, stop=100, step=1)

    # create noisy y
    y = x + np.random.normal(0, 1, 100)

    return (x, y)


class Optimizer(Enum):
    Adam = 1
    RMSProp = 2
    GradientDescent = 3
    StochasticGradientDescent = 4


class LinearRegression:
    def __init__(self, feature_count):
        self.feature_count = feature_count
        self._reset_weights()


    def _reset_weights(self):
        self.past_weights = []
        self.weights = np.zeros(self.feature_count)
        self.bias = 0


    def _save_weights(self):
        # we only have one weight so it's not that big of a deal...
        self.past_weights.append(self.weights[0])


    def train(self, X, y, optimizer=Optimizer.GradientDescent, epochs=1000, lr=1e-3):
        self._reset_weights()
        learning_curve = None

        match optimizer:
            case Optimizer.GradientDescent:
                print('Optimizing with Gradient Descent')
                learning_curve = _optimize_gd(X, y, epochs, lr, self)
            case Optimizer.Adam:
                print('Optimizing with Adam')
                learning_curve = _optimize_adam(X, y, [0.900, 0.999], epochs, lr, self)
            case Optimizer.RMSProp:
                pass
            case Optimizer.StochasticGradientDescent:
                print('Optimizing with Stochastic Gradient Descent')
                learning_curve = _optimize_sgd(X, y, epochs, lr, self)

        return learning_curve


    def obj_fn(self, X, y):
        return np.dot((self.predict(X) - y), X) / (np.shape(y) or 1)
    

    def obj_fn_b(self, X, y):
        return np.sum(self.predict(X) - y) / X.shape[0]


    def error(self, X, y):
        return np.sum(np.square(self.predict(X) - y)) / X.shape[0]


    def predict(self, X):
        return X @ self.weights + self.bias


class LogisticRegression:
    def __init__(self, feature_count):
        self._sigmoid_vectorized = np.vectorize(lambda z: self._sigmoid_overflow_proof(z))

        self.feature_count = feature_count
        self._reset_weights()


    def _reset_weights(self):
        self.past_weights = []
        self.weights = np.zeros(self.feature_count)
        self.bias = 0


    def train(self, X, y, optimizer=Optimizer.GradientDescent, epochs=1000, lr=1e-3):
        self._reset_weights()

        learning_curve = None

        match optimizer:
            case Optimizer.GradientDescent:
                print('Optimizing with Gradient Descent')
                learning_curve = _optimize_gd(X, y, epochs, lr, self)
            case Optimizer.Adam:
                print('Optimizing with Adam')
                learning_curve = _optimize_adam(X, y, [0.900, 0.999], epochs, lr, self)
            case Optimizer.RMSProp:
                pass
            case Optimizer.StochasticGradientDescent:
                print('Optimizing with Stochastic Gradient Descent')
                learning_curve = _optimize_sgd(X, y, epochs, lr, self)

        return learning_curve


    def obj_fn(self, X, y):
        cost = ((self.predict(X) - y).T @ X) / X.shape[0]
        return cost


    def _sigmoid_overflow_proof(self, z):
        if z < 0:
            return np.exp(z) / (1 + np.exp(z))
        else:
            return 1 / (1 + np.exp(-z))
    

    def error(self, X, y):
        y_hat = self.predict(X)
        return - (1 / X.shape[0]) * np.sum(y * np.log(y_hat) + (1 - y) * (np.log(1 - y_hat)))

    
    def predict(self, X):
        z = X @ self.weights + self.bias

        return self._sigmoid_vectorized(z)


    def _reset_weights(self):
        self.past_weights = []
        self.weights = np.zeros(self.feature_count)
        self.bias = 0


def _optimize_gd(X, y, epochs, lr, model):
    '''
    Optimizes model using gradient descent. 

    Parameters:
        X (np.array): training data features.
        y (np.array): training data target.
        epochs (int): number of iterations to train for.
        lr (float): the rate at which parameters change when performing an optimization step. 
        model (LogisticRegression or LinearRegression): the model who's weights will be optimized.

    Returns:
        A vector representing the loss through each training iteration...
    '''
    initial_mse = model.error(X, y)
    learning_curve = [initial_mse]

    for epoch in range(epochs):
        learning_curve.append(model.error(X, y))
        
        g_t = model.obj_fn(X, y)
        g_t_bias = np.sum(model.predict(X) - y) / X.shape[0]
        model.weights -= lr * g_t

        model.bias -= lr * g_t_bias

    return learning_curve


def _optimize_sgd(X, y, epochs, lr, model, batch_size=10):
    '''
    Optimizes model using gradient descent. 

    Parameters:
        X (np.array): training data features.
        y (np.array): training data target.
        epochs (int): number of iterations to train for.
        lr (float): the rate at which parameters change when performing an optimization step. 
        model (LogisticRegression or LinearRegression): the model that will be trained.
        batch_size (int): the number of training examples that will be sampled on each optimization step.

    Returns:
        A vector representing the loss through each training iteration...
    '''
    initial_mse = model.error(X, y)
    learning_curve = [initial_mse]

    for epoch in range(epochs):
        indexes = np.random.randint(low=0, high=X.shape[0], size=batch_size)

        x_i = np.zeros((batch_size, X.shape[1]))
        y_i = np.zeros(batch_size)

        for i in range(len(indexes)):
            x_i[i] = X[indexes[i]]
            y_i[i] = y[indexes[i]]

        mse = model.error(X, y)
        learning_curve.append(mse)
        
        g_t = model.obj_fn(x_i, y_i)
        g_t_bias = np.sum(model.predict(x_i) - y_i) / X.shape[0]
        model.weights -= lr * g_t

        model.bias -= lr * g_t_bias

    return learning_curve


def _optimize_adam(X, y, decay, epochs, lr, model):
    '''
    Optimizes model using ADAM optimization algorithm.

    Parameters:
        X (np.array): training data features.
        y (np.array): training data target.
        decay (tuple): decay rate for first and second moment vectors.
        epochs (int): number of iterations to train for.
        lr (float): todo

    Returns:
        A vector representing the loss through each training iteration...
    '''

    initial_mse = model.error(X, y)
    learning_curve = [initial_mse]

    # momentum vector...
    moment_m = 0
    moment_v = 0

    moment_m_bias = 0
    moment_v_bias = 0

    epsilon = 10 ** -8
    epoch = 0

    while epoch < epochs:
        epoch += 1

        g_t = model.obj_fn(X, y)
        g_t_bias = np.sum(model.predict(X) - y) / X.shape[0]
        
        moment_m = decay[0] * moment_m + (1 - decay[0]) * g_t
        moment_v = decay[1] * moment_v + (1 - decay[1]) * g_t ** 2

        moment_m_bias = decay[0] * moment_m_bias + (1 - decay[0]) * g_t_bias
        moment_v_bias = decay[1] * moment_v_bias + (1 - decay[1]) * g_t_bias ** 2

        moment_hat_m = moment_m / (1 - decay[0] ** epoch)
        moment_hat_v = moment_v / (1 - decay[1] ** epoch)

        moment_hat_m_bias = moment_m_bias / (1 - decay[0] ** epoch)
        moment_hat_v_bias = moment_v_bias / (1 - decay[1] ** epoch)

        model.weights = model.weights - lr * moment_hat_m / (np.sqrt(moment_hat_v) + epsilon)
        model.bias -= lr * moment_hat_m_bias / (np.sqrt(moment_hat_v_bias) + epsilon)

        error = np.sum((model.predict(X) - y) ** 2) / X.shape[0]
        learning_curve.append(error)

    return np.array(learning_curve)


if __name__ == '__main__':
    main()
