import torch
from torch import nn, optim
from torch.autograd import Variable
import json
from torch.utils.data import DataLoader, TensorDataset

class CNN(nn.Module):
    def __init__(self,):
        super().__init__()
        self.layer1_sentcent1 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3), #输出应为5*18*18
            nn.ReLU(inplace=True)
        )
        self.layer2_sentcent1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1)#输出应为5*17*17
        )
        self.layer1_sentcent2 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3),#同上
            nn.ReLU(inplace=True)
        )
        self.layer2_sentcent2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1)#同上
        )
        self.fc = nn.Sequential(
            nn.Linear(5*17*17*2, 64), #此时的输入为两个语句经过卷积层和池化层的特征
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 4)
        )

    def forward(self, x1, x2):
        x1 = self.layer1_sentcent1(x1)
        x1 = self.layer2_sentcent1(x1)
        x2 = self.layer1_sentcent2(x2)
        x2 = self.layer2_sentcent2(x2)
        x = torch.cat((x1, x2), 0).type(torch.FloatTensor)
        x = x.view(-1,5*17*17*2).cuda()
        o=self.fc(x)
        return o

learning_rate = 0.01

path='TextCNN\\train_newvector.json'
f = open(path,'r',encoding='utf8')
datajson=json.load(f)

querylist=torch.tensor([datajson[i]['query'] for i in range(len(datajson))])
titlelist=torch.tensor([datajson[i]['title'] for i in range(len(datajson))])
labellist=torch.tensor([int(datajson[i]['label']) for i in range(len(datajson))])
data=TensorDataset(querylist,titlelist,labellist)
data_loader=DataLoader(data,batch_size=1,shuffle=True)


model = CNN()
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


for num in range(64):
    epoch = 0
    #训练模型
    for query,title,label in data_loader:
        query=torch.unsqueeze(query, dim=1)
        title=torch.unsqueeze(title, dim=1)
        
        if torch.cuda.is_available():
            query = Variable(query.cuda())
            title = Variable(title.cuda())
            label = Variable(label.cuda())
        else:
            query = Variable(query)
            title = Variable(title)
            label = Variable(label)
        
        out = model(query,title)
        loss = criterion(out, label)
        print_loss = loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch+=1
        if epoch%1000 == 0:
            print('num: {},epoch: {}, loss: {:.4}'.format(num+1, epoch, loss.data.item()))

torch.save(model.state_dict(),'TextCNN\CNN.pt')

