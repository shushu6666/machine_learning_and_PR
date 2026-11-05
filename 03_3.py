from skimage import io
from skimage.color import rgb2lab, lab2rgb
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np
import os

path = 'style_transfer'
block_size = 1

def read_style_image(file_name, size=block_size):
    img = io.imread(file_name)
    # 统一到 Lab 空间，保证训练/预测特征一致
    img_lab = rgb2lab(img)

    fig = plt.figure()
    plt.imshow(img)
    plt.title('style image')
    plt.show()

    w, h = img_lab.shape[:2]
    X, Y = [], []
    for x in range(size, w - size):
        for y in range(size, h - size):
            # 特征：L 通道的局部窗口
            X.append(img_lab[x - size:x + size + 1,
                            y - size:y + size + 1, 0].flatten())
            # 标签：中心像素的 a、b 通道
            Y.append(img_lab[x, y, 1:])
    return np.array(X), np.array(Y)


def rebuild(img, size=block_size):
    fig = plt.figure()
    plt.imshow(img)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('content image')
    plt.show()

    img_lab = rgb2lab(img)
    w, h = img_lab.shape[:2]

    photo = np.zeros([w, h, 3])
    print('Constructing windows...')
    X = []
    for x in range(size, w - size):
        for y in range(size, h - size):
            window = img_lab[x - size:x + size + 1,
                             y - size:y + size + 1, 0].flatten()
            X.append(window)
    X = np.array(X)

    print('Predicting...')
    pred_ab = knn.predict(X).reshape(w - 2 * size, h - 2 * size, -1)

    # L 通道来自内容图，a/b 来自 KNN 预测
    photo[:, :, 0] = img_lab[:, :, 0]
    photo[size:w - size, size:h - size, 1:] = pred_ab
    photo = photo[size:w - size, size:h - size, :]
    return photo


X, Y = read_style_image(os.path.join(path, 'style.jpg'))
knn = KNeighborsRegressor(n_neighbors=4, weights='distance')
knn.fit(X, Y)

content = io.imread(os.path.join(path, 'input.jpg'))
new_photo = rebuild(content)

# Lab -> RGB 用于展示
new_photo = lab2rgb(new_photo)

fig = plt.figure(figsize=(10, 5))
plt.imshow(new_photo)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('stylized result')
plt.show()