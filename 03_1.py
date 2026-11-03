import matplotlib.pyplot as plt
import numpy as np
import os 



def distance(a,b):
    return np.sqrt(np.sum((a-b)**2))

class KNN:
    def __init__(self,k,label_num):
        self.k = k
        self.label_num = label_num 

    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train

    def get_knn_indices(self,x):
        dis = list(map(lambda a: distance(a,x),self.x_train))
        kn_indices = np.argsort(dis)
        knn_indices = kn_indices[:self.k]
        return knn_indices

    def get_label(self,x):
        knn_indices = self.get_knn_indices(x)
        label_statistic = np.zeros(shape=[self.label_num]) 
        for index in knn_indices:
            label = int(self.y_train[index])
            label_statistic[label] += 1
        return np.argmax(label_statistic)

    def predict(self,x_test):
         # 预测样本 test_x 的类别
        predicted_test_labels = np.zeros(shape=[len(x_test)], dtype=int)
        for i, x in enumerate(x_test):
            predicted_test_labels[i] = self.get_label(x)
        return predicted_test_labels

    
m_x = np.loadtxt('mnist_x')
m_y = np.loadtxt('mnist_y')

data = np .reshape(np.array(m_x[0],dtype=int),[28,28])
plt.figure()
plt.imshow(data,cmap = 'gray')
plt.title('MNIST sample')
plt.show()

ratio = 0.8
split = int(len(m_x) * ratio)
np.random.seed(0)
idx = np.random.permutation(np.arange(len(m_x)))
m_x = m_x[idx]
m_y = m_y[idx]
x_train,x_test = m_x[:split], m_x[split:]
y_train,y_test = m_y[:split], m_y[split:]

for k in range(1, 10):
    knn = KNN(k, label_num=10)
    knn.fit(x_train, y_train)
    predicted_labels = knn.predict(x_test)

    accuracy = np.mean(predicted_labels == y_test)
    print(f'K的取值为 {k}, 预测准确率为 {accuracy * 100:.1f}%')
