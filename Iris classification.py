import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
class Net(nn.Module):
    def __init__(self,in_num,out_num,hid_num):
        super(Net,self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_num,hid_num),
            nn.ReLU(),
            nn.Linear(hid_num,out_num)
        )
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.05)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self,x):
        return self.network(x)

    def train(self,x,y):
        out = self.forward(x)
        loss = self.loss_func(out,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print('loss = %.4f' % loss.item())

    def test(self,x):
        return self.forward()

if __name__ == '__main__':
    dataset = datasets.load_iris()
    input = torch.FloatTensor(dataset['data'])
    label = torch.LongTensor(dataset['target'])
    x_train, x_test, y_train, y_test = train_test_split(input, label, test_size=0.2)
    try:
        print("iris_model exist and have been loaded")
        mynet = torch.load('iris_model.pkl')
        output = mynet(input)
        pred_y = torch.max(output, 1)[1].numpy()
        sum = 0
        for i in range(len(label)):
            if pred_y[i] == label[i]:
                sum = sum + 1
        accuracy = float(sum / len(label))
        print('model accuracy = %d%% (testing on the whole dataset)' % (accuracy * 100))
    except:
        mynet = Net(4,10,3)
        accuracy = 0.0
        while accuracy < 0.9:
            for i in range (10000):
                mynet.train(x_train,y_train)
            output = mynet(x_test)
            pred_y = torch.max(output, 1)[1].numpy()
            sum=0
            for i in range(len(y_test)):
                if pred_y[i] == y_test[i]:
                    sum=sum+1
            accuracy = float(sum / len(y_test))
        torch.save(mynet, 'iris_model.pkl')
        print(mynet)
        print("The net have been saved")
        print('accuracy = %d%%' % (accuracy*100))








