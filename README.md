# 鸢尾花数据集分类--神经网络

#### 1.1 鸢尾花数据集介绍
iris数据集是用来给莺尾花做分类的数据集，每个样本包含了花萼长度、花萼宽度、花瓣长度、花瓣宽度四个特征，我们需要建立一个分类器，该分类器可通过样本的四个特征来来判断样本属于山鸢尾（Setosa）、变色鸢尾（Versicolour）还是维吉尼亚鸢尾（Virginica）中的哪一个，选择神经网络进行分类。

#### 1.2 思路流程
- 导入鸢尾花数据集
- 对数据集进行切分，分为训练集和测试集
- 搭建网络模型
- 训练网络
- 将所训练出的模型进行保存（准确率大于90%）

#### 1.3 网络模型

![](https://raw.githubusercontent.com/toygerse/MyProject/master/iris_network.png)

>采用sigmoid等函数，算激活函数时（指数运算），计算量大，反向传播求误差梯度时，求导涉及除法，计算量相对大，而采用Relu激活函数，整个过程的计算量节省很多，故采用Relu作为激活函数
#### 1.4 实现代码

导入所需要的的模块
```
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
```

神经网络类
```
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
```

引入数据集，并按照8:2切分训练集和测试集
```
dataset = datasets.load_iris()
input = torch.FloatTensor(dataset['data'])
label = torch.LongTensor(dataset['target'])
x_train, x_test, y_train, y_test = train_test_split(input, label, test_size=0.2)
```

如果存在已有训练好的网络则导入，并在总体数据集上测试其准确性
```
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
```

若不存在训练好的网络则进行训练，直到准确性大于90%后将其保存
```
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
```

#### 鸢尾花识别完整代码
```
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
```

