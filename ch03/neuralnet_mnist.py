# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    """
    784 = 28 * 28

    W1: (784, 50)
    b1: 50
    x ‧ W1 + b1 -> a1

    W2: (50, 100)
    b2: 100
    a1 ‧ W2 + b2 -> a2

    W3: (100, 10)
    b3: 10
    a2 ‧ W3 + b3 -> a3
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    """
    print('784 = 28 * 28')
    print('W1: ', len(W1), ',', len(W1[0]))
    print('b1: ', len(b1))
    print('x ‧ W1 + b1 -> a1')
    print('W2: ', len(W2), ',', len(W2[0]))
    print('b2: ', len(b2))
    print('a1 ‧ W2 + b2 -> a2')
    print('W3: ', len(W3), ',', len(W3[0]))
    print('b3: ', len(b3))
    print('a2 ‧ W3 + b3 -> a3')
    """

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 最も確率の高い要素のインデックスを取得
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x))) 
# Result stdout:
# Accuracy:0.9352