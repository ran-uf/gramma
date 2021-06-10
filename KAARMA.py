import numpy as np


class KAARMA:
    def __init__(self, ns, ny, a_s, a_u, u_type):
        self.ns = ns
        self.ny = ny
        self.a_s = a_s
        self.a_u = a_u
        self.S = (np.random.random((1, ns)) - 0.5) * 2
        self.phi = u_type[np.newaxis, :]
        self.A = (np.random.random((1, ns)) - 0.5) * 2

        self.II = np.zeros((ny, ns))
        self.II[:, ns - ny:] = np.eye(ny)

    def kernel_s(self, s1, s2):
        return np.exp(-self.a_s * np.sum(s1 - s2) ** 2)

    def kernel_u(self, u1, u2):
        return np.exp(-self.a_u * np.sum(u1 - u2) ** 2)

    def update(self, s, u, lr, dq):
        d_s = np.sum((self.S - s) ** 2, axis=1)
        d_u = (self.phi - u) ** 2
        dis = self.a_s * d_s + self.a_u * d_u.reshape(-1)
        index = np.argmin(dis)
        # print(dis[index])
        if dis[index] < dq:
            self.A[index] = self.A[index] + lr
        else:
            self.phi = np.concatenate((self.phi, np.array([u])[np.newaxis, :]), axis=0)
            self.A = np.concatenate((self.A, lr[np.newaxis, :]), axis=0)
            self.S = np.concatenate((self.S, s), axis=0)

    def test_one_sampe(self, x, y):
        s = np.zeros((1, self.ns))
        for f in x:
            di = self.S - s
            k_s = np.exp(-self.a_s * np.sum(di ** 2, axis=1))
            k_u = np.exp(-self.a_u * (self.phi - f) ** 2)
            ki = k_s * k_u.reshape(-1)
            s = self.A.T @ ki
        pred = self.II @ s
        # print(pred)
        if pred > 0.5:
            return np.sum((y - pred) ** 2), 1 == y
        else:
            return np.sum((y - pred) ** 2), 0 == y

    def train_3(self, x, y, lr, dq):
        L = 5
        for (u, yy) in zip(x, y):
            s0 = np.zeros((1, self.ns))
            length = u.shape[0]
            for t in range(length):
                t0 = max(t - L + 1, 0)
                gamma = []
                s = s0
                e = []
                # print(t)
                for i in range(t0, t + 1):
                    # print(i)
                    di = self.S - s

                    # compute k
                    k_s = np.exp(-self.a_s * np.sum(di ** 2, axis=1))
                    k_u = np.exp(-self.a_u * (self.phi - u[i]) ** 2).reshape(-1)
                    ki = k_s * k_u
                    s = self.A.T @ ki
                    ki = np.diag(ki.reshape(-1))
                    gamma_i = 2 * self.a_s * self.A.T @ ki @ di
                    gamma.append(gamma_i)
                    # todo
                    e.append(yy[i] - self.II @ s)

                elII = -e[0] @ self.II
                for i in range(1, len(gamma)):
                    ga = np.eye(self.ns)
                    for j in range(0, i):
                        ga = gamma[j + 1] @ ga
                    # todo
                    elII = elII - e[-1].T @ self.II @ ga

                self.update(s0, u[t0], -lr * elII @ np.eye(self.ns), dq)

                # update st0
                if t >= L - 1:
                    s0 = s0 + lr * elII
                # if t0 == 0:
                #     s0 = np.zeros((1, self.ns))
                # else:
                #     s0 = np.zeros((1, self.ns))
                #     for i in range(t0):
                #         ddi = self.S - s
                #         dk_s = np.exp(-self.a_s * np.sum(ddi ** 2, axis=1))
                #         dk_u = np.exp(-self.a_u * (self.phi - u[i]) ** 2)
                #         dki = dk_s * dk_u.reshape(-1)
                #         s0 = self.A.T @ dki
                #         s0 = s0[np.newaxis, :]

            # print(self.test_one_sampe(x[0], y[0]))
            # print('m:', self.A.shape[0], self.test_one_sampe(u, yy), yy)

    def train_1(self, x, y, lr, dq):
        num = x.shape[0]
        div = int(num * 0.7)
        for (u, d) in zip(x[:div], y[:div]):
            # generate s-1
            for i in [u.shape[0] - 1]:
                aaaaaaaaaa = 0
            # for i in range(u.shape[0]):
                s_p = []
                phi = []
                v = []
                ss = np.random.random((1, self.ns))
                for j in range(i + 1):
                    s_p.append(ss)
                    phi.append(u[j])
                    di = self.S - ss
                    k_s = np.exp(-self.a_s * np.sum(di ** 2, axis=1))[:, np.newaxis]
                    k_u = np.exp(-self.a_u * (self.phi - u[j]) ** 2)
                    ki = k_s * k_u
                    ss = self.A.T @ ki
                    ss = ss.T
                    # print(ki.tolist())
                    ki = np.diag(ki.reshape(-1))

                    gamma_i = 2 * self.a_s * self.A.T @ ki
                    if gamma_i.ndim == 1:
                        gamma_i = gamma_i[:, np.newaxis]
                    gamma_i = gamma_i @ di
                    if j == 0:
                        v.append(np.eye(self.ns))
                    else:
                        for index in range(len(v)):
                            v[index] = gamma_i @ v[index]
                        v.append(np.eye(self.ns))
                pred = self.II @ ss.T

                e = d - pred
                # if i == u.shape[0] - 1:
                #     print('\rerror:', e, ' m:', self.A.shape[0])

                # update weights
                for (s, uu, vv) in zip(s_p, phi, v):
                    m = self.A.shape[0]
                    # if m > 2000:
                    #    print('bug')
                    dis_s = np.sum((self.S - s) ** 2, axis=1)
                    dis_u = (self.phi - uu) ** 2
                    dis_u = dis_u.reshape(-1)
                    dis = self.a_s * dis_s + self.a_u * dis_u
                    dis = dis[:m]
                    index = np.argmin(dis)
                    a = self.II @ vv
                    a = a.T @ e

                    if dis[index] < dq:
                        a = a.reshape(-1)
                        self.A[index] = self.A[index] + lr * a
                    else:
                        # print(self.S.shape, s.shap
                        self.A = np.concatenate((self.A, lr * a.T), axis=0)
                        self.phi = np.concatenate((self.phi, np.array([uu])[np.newaxis, :]), axis=0)
                        self.S = np.concatenate((self.S, s), axis=0)

                    print('\rprogress: %03f dis: ' % (i / u.shape[0]), dis[index], ' m: ', m,  end=' ')

            # print(self.test_one_sampe(u, d))

        loss_train = []
        num_train = 0
        for (train_x, train_y) in zip(x[:div], y[:div]):
            ls, count = self.test_one_sampe(train_x, train_y)
            loss_train.append(ls)
            num_train = num_train + count

        loss_test = []
        num_test = 0
        for (test_x, test_y) in zip(x[div:], y[div:]):
            ls, count = self.test_one_sampe(test_x, test_y)
            loss_test.append(ls)
            num_test = num_test + count

        print('\rloss_train: ', np.mean(loss_train), ' acc_train:', num_train / len(loss_train), ' loss_test: ', np.mean(loss_test), ' acc_test', num_test / len(loss_test))
