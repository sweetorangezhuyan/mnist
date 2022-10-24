'''
1. data processing
'''
import pandas as pd
import numpy as np
train_feature=pd.read_csv('train.csv',header=None)
train_result=pd.read_csv('train_result.csv',header=None)
test_data=pd.read_csv('test.csv',header=None)
ma_train_feature=np.array(train_feature.values[1:,:-1],dtype=float) # 50000*1568
ma_train_labels=np.array(train_result.values[1:,-1],dtype=int)

ma_test_feature=np.array(test_data.values[1:,:-1],dtype=float)# 10000*1568

sumcol=np.sum(ma_train_feature,axis=0)
cols=[ind for ind,num in enumerate(sumcol) if num==0.0]## record the sum==0.0 and delete

ma_train_feature=np.delete(ma_train_feature,cols,axis=1)
ma_test_feature=np.delete(ma_test_feature,cols,axis=1)

'''
2. the class of LogisticRegression
'''
class LogisticRegression_mutli_class():
    def __init__(self,max_iter=500,alpha=0.25):
        self.penalty='l2'
        self.max_iter=max_iter
        self.alpha=alpha
    def initialize_with_zeros(self):
        self.w = np.zeros((self.n_features,self.n_classes))
         
    def softmax(self,features):
        row_max = np.max(features, axis=1).reshape(-1, 1)
        features -= row_max
        features_exp = np.exp(features)
        s = features_exp / np.sum(features_exp, axis=1, keepdims=True)
        return s
    def predict(self,X_test):
        pro=self.softmax(np.dot(X_test,self.w))
        return np.argmax(pro,axis=1)
    def loss_function(self,X,y_one_hot):
        loss=0
        soft=self.softmax(np.dot(X,self.w))
        n=len(y_one_hot)
        loss=(-1/n)*np.sum(np.sum(y_one_hot*np.log(soft),axis=1))
        #print(soft.shape)
        grad=(-1/n)*np.dot(X.T,(y_one_hot-soft))
        return loss,grad
    def fit(self,X,y):
        self.classes=np.unique(y)      
        self.n_classes=len(self.classes)
        y_one_hot=np.eye(self.n_classes)[y]
        self.n_features=X.shape[1]
        self.initialize_with_zeros()
        for i in range(self.max_iter):
            loss,grad=self.loss_function(X,y_one_hot)
            #print(self.alpha*grad)
            self.w=self.w-(self.alpha*grad)
            if i %10==0:
                print(loss)
'''
3. the train of model
'''
model=LogisticRegression_mutli_class(max_iter=600,alpha=0.25)
model.fit(ma_train_feature,ma_train_labels)

'''
4. predict the test data
'''
pre_test=model.predict(ma_test_feature)
def save_csv(data,file_name):
    pre_list=[]
    for ind, num in enumerate(data):
        pre_list.append([ind,num])
    name=['Index','Class']
    test=pd.DataFrame(columns=name,data=pre_list)
    test.head()
    test.to_csv(file_name,index=False)
save_csv(pre_test,'logistic_regression.csv')
