import numpy as np
from filter import generate_data


def identity(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    a = np.exp(x)
    b = np.exp(-x)
    return (a - b) / (a + b)


class RNN(object):
    def __init__(self, dim_hidden):

        self.dim_hidden = dim_hidden

        self.w0 = 2 * np.random.random((1, dim_hidden)) - 1
        self.w1 = 2 * np.random.random((dim_hidden, dim_hidden)) - 1
        self.w2 = 2 * np.random.random((dim_hidden, 1)) - 1

        self.acivate = sigmoid

    def forward(self, x):
        length = x.shape[0]
        y1 = np.zeros((length + 1, self.dim_hidden))

        for i in range(length):
            y1[i + 1, :] = self.acivate(np.dot(x[i], self.w0) + np.dot(y1[i, :], self.w1))
        y2 = self.acivate(self.w2.T @ y1[-1, :])

        return y2

    def train_step(self, x, y, lr):
        length = x.shape[0]
        y1 = np.zeros((length + 1, self.dim_hidden))

        # e = np.zeros_like(y, dtype=np.float64)
        # dj_dy2 = np.zeros_like(y, dtype=np.float64)
        for i in range(length):
            y1[i + 1, :] = self.acivate(np.dot(x[i], self.w0) + np.dot(y1[i, :], self.w1))
        y2 = self.acivate(self.w2.T @ y1[-1, :])
        e = y - y2

        dj_dy2 = -e * y2 * (1 - y2)
        # print(abs(e))
        dj_dw0 = np.zeros_like(self.w0)
        dj_dw1 = np.zeros_like(self.w1)
        dj_dw2 = np.zeros_like(self.w2)

        # dj_dw1 = np.zeros((self.dim_hidden, self.dim_hidden))
        delta_future_y1 = np.zeros(self.dim_hidden)

        for i in range(length, 0, -1):
            if i == length:
                delta_y1 = np.atleast_2d(dj_dy2).dot(self.w2.T) * y1[i, :] * (1 - y1[i, :])
            else:
                delta_y1 = (delta_future_y1.dot(self.w1.T)) * y1[i, :] * (1 - y1[i, :])

            dj_dw0 += np.atleast_2d(x[i - 1]).T.dot(delta_y1)
            dj_dw1 += np.atleast_2d(y1[i - 1, :]).T.dot(delta_y1)

            delta_future_y1 = delta_y1

        dj_dw2 += np.atleast_2d(y1[-1, :]).T.dot(np.atleast_2d(dj_dy2))

        self.w0 = self.w0 - lr * dj_dw0
        self.w1 = self.w1 - lr * dj_dw1
        self.w2 = self.w2 - lr * dj_dw2

        dj_dw0 *= 0
        dj_dw1 *= 0
        dj_dw2 *= 0

    def train(self, x, y, lr):
        for (xx, yy) in zip(x, y):
            self.train_step(xx, yy[0], lr)
            # losss = self.loss(xx, yy)
            # acc = self.accuracy(x, y)
            # print(np.mean(losss))

    def loss(self, x, y):
        ls = list()
        res = list()
        for (xx, yy) in zip(x, y):
            r = self.forward(xx)
            ls.append(yy[-1] - r)
            res.append((r >= 0.5) == yy[-1])
        return np.mean(np.array(ls) ** 2), np.mean(res)


if __name__ == '__main__':
    from tomita4 import generate_tomita4

    model = RNN(32)
    x_train = []
    y_train = []
    for i in [6, 7, 8, 9, 10]:
        x, y = generate_tomita4(20, i)
        for xx, yy in zip(x, y):
            x_train.append(xx)
            y_train.append(yy)
    x_train = np.array(x_train, dtype='object')
    y_train = np.array(y_train, dtype='object')

    x_test, y_test = generate_tomita4(40, 10)
    # index = np.random.permutation(x_train.size)
    # x_train = x_train[index]
    # y_train = y_train[index]
    # x_train = x_train[1][np.newaxis, :]
    # y_train = y_train[1][np.newaxis, :]
    # x_train = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 0, 1]])
    # y_train = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]])
    losses_train = list()
    losses_test = list()
    acc_train = list()
    acc_test = list()
    for step in range(1000):
        # index = np.random.permutation(x_train.size)
        model.train(x_train, y_train, 0.01)
        a, acc_a = model.loss(x_train, y_train)
        b, acc_b = model.loss(x_test, y_test)
        print('loss_train:', a, 'loss_test:', b)
        losses_train.append(a)
        losses_test.append(b)
        acc_train.append(acc_a)
        acc_test.append(acc_b)

    # acc = model.accuracy(x_train, y_train)
    # print(np.mean(losss), acc)

    import matplotlib.pyplot as plt

    plt.plot(losses_train, label='loss_train')
    plt.plot(losses_test, label='loss_test')
    plt.legend()
    plt.show()

    plt.plot(acc_train, label='accuracy_train')
    plt.plot(acc_test, label='accuracy_test')
    plt.legend()
    plt.show()
