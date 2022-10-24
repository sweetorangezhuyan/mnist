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
2. initial and train model
'''
from sklearn.ensemble import RandomForestClassifier
randf=RandomForestClassifier()
randf.fit(ma_train_feature,ma_train_labels)

'''
3.redict the test data
'''
pre_test=randf.predict(ma_test_feature)
def save_csv(data,file_name):
    pre_list=[]
    for ind, num in enumerate(data):
        pre_list.append([ind,num])
    name=['Index','Class']
    test=pd.DataFrame(columns=name,data=pre_list)
    test.head()
    test.to_csv(file_name,index=False)
save_csv(pre_test,'random_forest.csv')