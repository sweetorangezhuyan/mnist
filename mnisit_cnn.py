'''
1. data processing
'''
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch 
import torch.nn as nn
train_feature=pd.read_csv('train.csv',header=None)
train_result=pd.read_csv('train_result.csv',header=None)
test_data=pd.read_csv('test.csv',header=None)
x_train=np.array(train_feature.values[1:,:-1],dtype=float) # 50000*1568
y_train=np.array(train_result.values[1:,-1],dtype=int)
x_test=np.array(test_data.values[1:,:-1],dtype=float) 

'''
2.create model
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(19968, 128)
        self.fc2 = nn.Linear(128, 19)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

'''
3. initial model, loss, optim
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cnn=Net()
loss_fn = nn.CrossEntropyLoss()     
optimizer = torch.optim.Adadelta(model_cnn.parameters(), lr=3)
epoch=50

'''
4. training
'''
bs=100
rand_inds = np.random.permutation(len(y_train))
rand_inds = rand_inds[:int(len(rand_inds)/bs)*bs]
rand_inds = np.reshape(rand_inds,[-1, bs])
rand_inds.shape
model_cnn.train()
for i in range(epoch):
    for ind in range(len(rand_inds)):
        inds=rand_inds[ind]
        x_now=x_train[inds]
        x_now=x_now.reshape((-1,56,28))
        y_now=y_train[inds]
        x_now=torch.Tensor(x_now).to(device)
        x_now=torch.unsqueeze(x_now,dim=1)
        y_now=torch.Tensor(y_now).long().to(device)
        x_net=model_cnn(x_now)
        loss=loss_fn(x_net,y_now)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ind%200==0 and i%5==0:
            print(loss)
'''
4. testing and save
'''
model_cnn.eval()
correct=0
nums=int(len(x_test)/bs)
result=np.zeros((len(x_test)))
for ind in range(nums):
    x_now=x_test[ind*bs:(ind+1)*bs]
    x_now=x_now.reshape((-1,56,28))
    x_now=torch.Tensor(x_now).to(device)
    x_now=torch.unsqueeze(x_now,dim=1)
    x_net=model_cnn(x_now)
    x_net=x_net.argmax(1)
    pre=x_net.detach().cpu().numpy()
    result[ind*bs:(ind+1)*bs]=pre
def save_csv(pre_y,filename):
    pre_list=[]
    for ind, num in enumerate(pre_y):
        pre_list.append([ind,int(num)])
    name=['Index','Class']
    test=pd.DataFrame(columns=name,data=pre_list)
    test.head()
    test.to_csv(filename,index=False)
save_csv(result,'cnn_3.csv')