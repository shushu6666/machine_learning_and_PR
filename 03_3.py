from skimage import io
from skimage.color import rgb2lab,lab2rgb
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import os

path = 'style_transfer'

data_dir = os.path.join(path,'vangogh')
fig = plt.figure(figsize=(16,5))
for i,file in enumerate(np.sort(os.listdir(data_dir))[:3]):
    img = io.imread(os.path.join(data_dir,file))
    ax = fig.add_subplot(1,3,i+1)
    ax.imshow(img)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_title(file)
plt.show()

block_size = 1

def read_style_image(file_name,size = block_size):
    img = io.imread(file_name)
    fig = plt.figure()
    plt.imshow(img)
    w,h = img.shape[:2]

    X = []
    Y = []
    for x in range(size,w - size):
        for y in range(size,h-size):
            X.append(img[x - size:x + size+1,y-size:y + size+1,0].flatten())
            Y.append(img[x,y,1:])
    return X,Y


def rebuild(img,size = block_size):
    fig = plt.figure()
    plt.imshow(img)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.show()

    img = rgb2lab(img)
    w,h = img.shape[:2]

    photo = np.zeros([w,h,3])
    print('Constructing windows...')
    X = []
    for x in range(size,w - size):
        for y in range(size,h-size):
            window = img[x - size:x + size+1,y-size:y + size+1,0].flatten()
            X.append(window)
    X= np.array(X)
    print('Predicting...')
    pred_ab = knn.predict(X).reshape(w - 2*size,h - 2*size,-1)
    photo[:,:,0] = img[:,:,0]
    photo[size:w-size,size:h-size,1:] = pred_ab
    photo = photo[size:w-size,size:h-size,:]
    return photo

X,Y = read_style_image(os.path.join(path,'style.jpg'))
knn = KNeighborsRegressor(n_neighbors=4,weights='distance')
knn.fit(X,Y)
content = io.imread(os.path.join(path, 'input.jpg'))
new_photo = rebuild(content)
# 为了展示图像，我们将其再转换为RGB表示
new_photo = lab2rgb(new_photo)

fig = plt.figure()
plt.imshow(new_photo)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.show()

