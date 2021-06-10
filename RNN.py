import numpy as np
from filter import generate_data


int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]


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
        np.random.seed(0)
        self.w0 = 2 * np.random.random((2, dim_hidden)) - 1
        np.random.seed(1)
        self.w1 = 2 * np.random.random((dim_hidden, dim_hidden)) - 1
        np.random.seed(2)
        self.w2 = 2 * np.random.random((dim_hidden, 1)) - 1

        self.acivate = sigmoid

    def forward(self, x):
        length = x.shape[0]
        y1 = np.zeros((length + 1, self.dim_hidden))
        out = np.zeros(x.shape[0])
        for i in range(length):
            y1[i + 1, :] = self.acivate(np.dot(x[i], self.w0) + np.dot(y1[i, :], self.w1))
            # d_a[i] = a[i] * (1 - a[i])
            y2 = self.acivate(self.w2.T @ y1[i + 1, :])
            out[i] = y2
        return out

    def train_step(self, x, y, lr):
        length = x.shape[0]
        y1 = np.zeros((length + 1, self.dim_hidden))

        # a_y1 = np.zeros((length, self.dim_hidden))
        # a_y2 = np.zeros((length, self.dim_hidden))
        e = np.zeros_like(y, dtype=np.float64)
        dj_dy2 = np.zeros_like(y, dtype=np.float64)
        for i in range(length):
            y1[i + 1, :] = self.acivate(np.dot(x[i], self.w0) + np.dot(y1[i, :], self.w1))
            # d_a[i] = a[i] * (1 - a[i])
            y2 = self.acivate(self.w2.T @ y1[i + 1, :])
            e[i] = y[i] - y2

            dj_dy2[i] = -e[i] * y2 * (1 - y2)
        print(sum(abs(e)))
        dj_dw0 = np.zeros_like(self.w0)
        dj_dw1 = np.zeros_like(self.w1)
        dj_dw2 = np.zeros_like(self.w2)

        # dj_dw1 = np.zeros((self.dim_hidden, self.dim_hidden))
        delta_future_y1 = np.zeros(self.dim_hidden)

        for i in range(length, 0, -1):
            if i == length:
                delta_y1 = np.atleast_2d(dj_dy2[i - 1]).dot(self.w2.T) * y1[i, :] * (1 - y1[i, :])
            else:
                # aa = delta_future_y1.dot(self.w1.T)
                # bb = np.atleast_2d(dj_dy2[i - 1]).dot(self.w2.T) * y1[i, :] * (1 - y1[i, :])
                delta_y1 = (delta_future_y1.dot(self.w1.T) + np.atleast_2d(dj_dy2[i - 1]).dot(self.w2.T)) * y1[i, :] * (1 - y1[i, :])

            dj_dw0 += np.atleast_2d(x[i - 1]).T.dot(delta_y1)
            dj_dw1 += np.atleast_2d(y1[i - 1, :]).T.dot(delta_y1)
            dj_dw2 += np.atleast_2d(y1[i, :]).T.dot(np.atleast_2d(dj_dy2[i - 1]))

            delta_future_y1 = delta_y1

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


if __name__ == '__main__':
    from tomita4 import generate_tomita4

    model = RNN(16)
    x_train = []
    y_train = []
    for i in [10]:
        x, y = generate_tomita4(40, i)
        for xx, yy in zip(x, y):
            x_train.append(xx)
            y_train.append(yy)
    x_train = np.array(x_train, dtype='object')
    y_train = np.array(y_train, dtype='object')

    # index = np.random.permutation(x_train.size)
    # x_train = x_train[index]
    # y_train = y_train[index]
    # x_train = x_train[1][np.newaxis, :]
    # y_train = y_train[1][np.newaxis, :]
    # x_train = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 0, 1]])
    # y_train = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]])

    for step in range(10000):
        # index = np.random.permutation(x_train.size)
        a_int = np.random.randint(largest_number / 2)  # int version
        a = int2binary[a_int]  # binary encoding

        b_int = np.random.randint(largest_number / 2)  # int version
        b = int2binary[b_int]  # binary encoding

        # true answer
        c_int = a_int + b_int
        c = int2binary[c_int]

        x = np.vstack([a, b]).T
        xx = x[::-1, :]
        model.train_step(xx, c[::-1], 0.1)
        # losss = model.loss(x_train, y_train)
    # acc = model.accuracy(x_train, y_train)
    # print(np.mean(losss), acc)

