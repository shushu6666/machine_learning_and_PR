from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import  ListedColormap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))

class KNN:
    def __init__(self, k, label_num):
        self.k = k
        self.label_num = label_num # 类别的数量

    def fit(self, x_train, y_train):
        # 在类中保存训练数据
        self.x_train = x_train
        self.y_train = y_train

    def get_knn_indices(self, x):
        # 获取距离目标样本点最近的K个样本点的标签
        # 计算已知样本的距离
        dis = list(map(lambda a: distance(a, x), self.x_train)) 
        # 按距离从小到大排序，并得到对应的下标
        knn_indices = np.argsort(dis) 
        # 取最近的K个
        knn_indices = knn_indices[:self.k] 
        return knn_indices

    def get_label(self, x):
        # 对KNN方法的具体实现，观察K个近邻并使用np.argmax获取其中数量最多的类别
        knn_indices = self.get_knn_indices(x)
        # 类别计数
        label_statistic = np.zeros(shape=[self.label_num]) 
        for index in knn_indices:
            label = int(self.y_train[index])
            label_statistic[label] += 1
        # 返回数量最多的类别
        return np.argmax(label_statistic) 

    def predict(self, x_test): 
        # 预测样本 test_x 的类别
        predicted_test_labels = np.zeros(shape=[len(x_test)], dtype=int)
        for i, x in enumerate(x_test):
            predicted_test_labels[i] = self.get_label(x)
        return predicted_test_labels

    
data = np.loadtxt('gauss.csv',delimiter=',')
x_train = data[:,:2]
y_train = data[:,2]
print('数据集大小：',len(x_train))

plt.figure()
plt.scatter(x_train[y_train ==0,0],x_train[y_train ==0,1],c='blue',marker='o')
plt.scatter(x_train[y_train ==1,0],x_train[y_train ==1,1],c='red',marker='x')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()



step = 0.02
x_min,x_max = np.min(x_train[:,0]) -1,np.max(x_train[:,0])+1
y_min,y_max = np.min(x_train[:,1]) -1,np.max(x_train[:,1])+1

xx,yy = np.meshgrid(np.arange(x_min,x_max,step),np.arange(y_min,y_max,step))
grid_data = np.concatenate([xx.reshape(-1,1),yy.reshape(-1,1)],axis=1)
fig = plt.figure(figsize = (16,4.5))
ks = [1,3,10]
cmap_light = ListedColormap(['royalblue','lightcoral'])

for i,k in enumerate(ks):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    z = knn.predict(grid_data)
    ax = fig.add_subplot(1, 3, i + 1)
    ax.pcolormesh(xx, yy, z.reshape(xx.shape), cmap=cmap_light, alpha=0.7)
    ax.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], c='blue', marker='o')
    ax.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], c='red', marker='x')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_title(f'K = {k}')
plt.savefig('debug_plot.png', dpi=100)
plt.show()



