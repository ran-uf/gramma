import numpy as np
import torch
from tomita4 import generate_tomita4


class RNN(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(RNN, self).__init__()
        self.rnn = torch.nn.RNN(1, hidden_dim, 1, batch_first=True, bias=False)
        self.linear = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        # for time in range(r_out.size(1)):  # r_out.size=[1,10,32]即将一个长度为10的序列的每个元素都映射到隐藏层上.
        #     outs.append(self.linear(r_out[:, time, :]))  # 依次抽取序列中每个单词,将之通过全连接层并输出.r_out[:, 0, :].size()=[1,32] -> [1,1]
        # return torch.stack(outs, dim=1), h_state
        return self.linear(r_out[:, r_out.size(1) - 1, :]), h_state


x_org, y_org = generate_tomita4(1000, 10)
# x_org = (x_org - 0.5) * 2
# y_org = (y_org - 0.5) * 2
x = torch.from_numpy(x_org[:, :, np.newaxis].astype(np.float32))
y = torch.from_numpy(y_org[:, :, np.newaxis].astype(np.float32))
model = RNN(16)
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
h_state = None

ls = []
for step in range(2000):
    pred, h_state = model(x, h_state)
    h_state = h_state.data
    loss = loss_func(pred, y[:, 0, :])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ls.append(loss.data)

pred = pred.data.numpy()[:, 0]
import matplotlib.pyplot as plt

plt.plot(ls)
plt.show()