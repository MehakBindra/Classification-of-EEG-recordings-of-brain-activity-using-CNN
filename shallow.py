import numpy as np
import torch
import scipy.io as sio

from deep import net


def get_data(subject, training, PATH):
    # load data from .mat files
    NO_channels = 22
    NO_tests = 6 * 48
    Window_Length = 7 * 250
    # 250 Hz is the sampling rate

    class_return = np.zeros(NO_tests)
    data_return = np.zeros((NO_tests, Window_Length, NO_channels))

    NO_valid_trial = 0
    if training:
        a = sio.loadmat(PATH + 'A0' + str(subject) + 'T.mat')
    else:
        a = sio.loadmat(PATH + 'A0' + str(subject) + 'E.mat')
    a_data = a['data']
    for ii in range(0, a_data.size):
        a_data1 = a_data[0, ii]
        a_data2 = [a_data1[0, 0]]
        a_data3 = a_data2[0]
        a_X = a_data3[0]
        a_trial = a_data3[1]
        a_y = a_data3[2]
        a_artifacts = a_data3[5]
        for trial in range(0, a_trial.size):
            if a_artifacts[trial] == 0:
                data_return[NO_valid_trial, :, :] = \
                    a_X[int(a_trial[trial]):(int(a_trial[trial]) + Window_Length), :22]
                class_return[NO_valid_trial] = int(a_y[trial])
                NO_valid_trial += 1

    return data_return[0:NO_valid_trial, :, :], class_return[0:NO_valid_trial]


class Expression(torch.nn.Module):
    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if (hasattr(self.expression_fn, 'func') and
                hasattr(self.expression_fn, 'kwargs')):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__,
                str(self.expression_fn.kwargs))
        elif hasattr(self.expression_fn, '__name__'):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (self.__class__.__name__ + '(' +
                'expression=' + str(expression_str) + ')')


class ShallowCNN(torch.nn.Module):
    # Proposed CNN architecture with less number of layers

    def __init__(self):
        super(ShallowCNN, self).__init__()

        self.conv_time = torch.nn.Conv2d(1, 40, (25, 1), stride=1)
        self.conv_spat = torch.nn.Conv2d(40, 40, (1, 22), stride=1, bias=False)
        self.batch0 = torch.nn.BatchNorm2d(40, momentum=0.1, affine=True)
        self.sq0 = Expression(square)
        self.pool0 = torch.nn.AvgPool2d(kernel_size=(75, 1), stride=(15, 1))
        self.drop1 = torch.nn.Dropout(p=0.5)

        self.final_conv_length = 111

        self.classifier = torch.nn.Conv2d(40, 4, (self.final_conv_length, 1), bias=True)
        self.soft = torch.nn.LogSoftmax(dim=1)

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.conv_time.weight, gain=1)
        torch.nn.init.constant_(self.conv_time.bias, 0)
        torch.nn.init.xavier_uniform_(self.conv_spat.weight, gain=1)
        torch.nn.init.constant_(self.batch0.weight, 1)
        torch.nn.init.constant_(self.batch0.bias, 0)

        torch.nn.init.xavier_uniform_(self.classifier.weight, gain=1)
        torch.nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x, isTraining=True):

        x = torch.from_numpy(np.array([[x]], dtype=np.float32))
        x = self.conv_time(x)
        x = self.conv_spat(x)
        if isTraining:
            x = self.batch0(x)
        x = self.sq0(x)
        x = (self.pool0(x))
        if isTraining:
            x = self.drop1(x)

        x = self.classifier(x)
        x = self.soft(x)
        x = x[:, :, :, 0]
        x = x[:, :, 0]
        x = x[0, :]
        return x


def square(x):
    return x * x


def crossEntropy(preds, labels):
    # loss function
    loss = -preds[int(labels.item()) - 1]
    loss = loss / 4
    return loss


def train(net, optimizer, criterion, x_train, y_train, max_epochs=10):
    # train function
    train_losses = []

    for epoch in range(max_epochs):
        running_loss = 0
        cnt = 0
        for ix in range(x_train.shape[0]):
            optimizer.zero_grad()
            preds = net(x_train[ix])
            loss = criterion(preds, y_train[ix])
            loss.backward()
            optimizer.step()

            cnt += 1
            running_loss += loss.item()

        print("Epoch {}: Training Loss {}".format(epoch + 1, running_loss / cnt))
        train_losses.append(running_loss / cnt)

    return train_losses


X_train, Y_train = get_data('1', True, 'data/')

for i in range(2, 10):
    tempx, tempy = get_data(str(i), True, 'data/')
    X_train = np.append(X_train, tempx, axis=0)
    Y_train = np.append(Y_train, tempy, axis=0)

for i in range(2, 6):
    tempx, tempy = get_data(str(i), False, 'data/')
    X_train = np.append(X_train, tempx, axis=0)
    Y_train = np.append(Y_train, tempy, axis=0)

for i in range(8, 10):
    tempx, tempy = get_data(str(i), False, 'data/')
    X_train = np.append(X_train, tempx, axis=0)
    Y_train = np.append(Y_train, tempy, axis=0)

Y_train = torch.from_numpy(Y_train)
net2 = ShallowCNN()
opti = torch.optim.Adam(net2.parameters(), lr=1e-5)
net2.initialize()

t2 = train(net2, opti, crossEntropy, X_train, Y_train)

X_test, Y_test = get_data('1', False, 'data/')
tempx, tempy = get_data('7', False, 'data/')
X_test = np.append(X_test, tempx, axis=0)
Y_test = np.append(Y_test, tempy, axis=0)

Y_test = torch.from_numpy(Y_test)

count = 0  # calculating accuracy
for i in range(X_test.shape[0]):
    pred = net2(X_test[i], False).detach().numpy()
    if (pred.argmax(0) + 1) == int(Y_test[i].item()):
        count += 1

accu2 = count / X_test.shape[0]
print(accu2)
