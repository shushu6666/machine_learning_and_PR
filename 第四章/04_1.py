import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
# 该函数每次返回大小为batch_size的批量
# x和y分别为输入和标签
# 若shuffle = True，则每次遍历时会将数据重新随机划分
def batch_generator(x, y, batch_size, shuffle=True):
    # 批量计数器
    batch_count = 0
    if shuffle:
        # 随机生成0到len(x)-1的下标
        idx = np.random.permutation(len(x))
        x = x[idx]
        y = y[idx]
    while True:
        start = batch_count * batch_size
        end = min(start + batch_size, len(x))
        if start >= end:
            # 已经遍历一遍，结束生成
            break
        batch_count += 1
        yield x[start: end], y[start: end]

def SGD(num_epoch,learning_rate,batch_size):
    X = np.concatenate([x_train,np.ones((len(x_train),1))],axis=-1)
    X_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=-1)
    theta = np.random.normal(size=X.shape[1])

    train_losses = []
    test_losses = []
    for i in range(num_epoch):
        batch_g = batch_generator(X, y_train, batch_size, shuffle=True)
        train_loss = 0
        for x_batch, y_batch in batch_g:
            # 计算梯度
            grad = x_batch.T @ (x_batch @ theta - y_batch)
            # 更新参数
            theta = theta - learning_rate * grad / len(x_batch)
            # 累加平方误差
            train_loss += np.square(x_batch @ theta - y_batch).sum()
        # 计算训练和测试误差
        train_loss = np.sqrt(train_loss / len(X))
        train_losses.append(train_loss)
        test_loss = np.sqrt(np.square(X_test @ theta - y_test).mean())
        test_losses.append(test_loss)
        print('回归系数：', theta)
    return theta, train_losses, test_losses

    
lines = np.loadtxt('USA_Housing.csv',delimiter = ',',dtype = 'str')
header = lines[0]
lines = lines[1:].astype(float)
print('数据特征：',','.join(header[:-1]))
print('数据标签：',header[-1])
print('数据总条数：',len(lines))

ratio = 0.8
split = int(len(lines) * ratio)
np.random.seed(0)
lines = np.random.permutation(lines)
train,test = lines[:split],lines[split:]

scaler = StandardScaler()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

x_train,y_train = train[:,:-1],train[:,-1].flatten()
x_test,y_test = test[:,:-1],test[:,-1].flatten()

X = np.concatenate([x_train,np.ones((len(x_train),1))],axis=-1)
theta = np.linalg.inv(X.T @ X) @ X.T @ y_train
print('线性回归模型参数：',theta)

X_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=-1)
y_pred = X_test @ theta


rmse_loss = np.sqrt(np.square(y_test - y_pred).mean())
print('RMSE：', rmse_loss)
num_epoch = 20
learning_rate = 0.01
batch_size = 32
# 设置随机种子
np.random.seed(0)

_, train_losses, test_losses = SGD(num_epoch, learning_rate, batch_size)
    
# 将损失函数关于运行次数的关系制图，可以看到损失函数先一直保持下降，之后趋于平稳
plt.plot(np.arange(num_epoch), train_losses, color='blue', 
    label='train loss')
plt.plot(np.arange(num_epoch), test_losses, color='red', 
    ls='--', label='test loss')
# 由于epoch是整数，这里把图中的横坐标也设置为整数
# 该步骤也可以省略
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.show()
_, loss1, _ = SGD(num_epoch=num_epoch, learning_rate=0.1, 
    batch_size=batch_size)
_, loss2, _ = SGD(num_epoch=num_epoch, learning_rate=0.001, 
    batch_size=batch_size)
plt.plot(np.arange(num_epoch), loss1, color='blue', 
    label='lr=0.1')
plt.plot(np.arange(num_epoch), train_losses, color='red', 
    ls='--', label='lr=0.01')
plt.plot(np.arange(num_epoch), loss2, color='green',
    ls='-.', label='lr=0.001')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.show()
_, loss3, _ = SGD(num_epoch=num_epoch, learning_rate=1.5, batch_size=batch_size)
print('最终损失：', loss3[-1])
plt.plot(np.arange(num_epoch), np.log(loss3), color='blue', label='lr=1.5')
plt.xlabel('Epoch')
plt.ylabel('log RMSE')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.show()

