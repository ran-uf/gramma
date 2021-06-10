import pickle
import numpy as np
from KAARMA import KAARMA
from tomita4 import generate_tomita4


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    x, y = generate_tomita4(40, 10)
    # train_x = []
    # train_y = []
    # for i in range(0, x.shape[0], 2):
    #     train_x.append(x[i:i + 2, :].reshape(-1))
    #     train_y.append(y[i:i + 2, :].reshape(-1))
    # xx = x.reshape(-1)
    # yy = np.array([y.tolist()] * 20).T.reshape(-1)

    model = KAARMA(5, 1, 2, 2, np.array([x[0, 0]]))

    for i in range(100):
        model.train_3(x, y, .1, 0.001)

        loss = []
        res = []
        for (xxx, yyy) in zip(x, y[:, 0]):
            l, r = model.test_one_sampe(xxx, yyy)
            loss.append(l)
            res.append(r)
        print(np.mean(loss), np.mean(res))

