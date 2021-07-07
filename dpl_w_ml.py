import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib as mpl
import mlflow as ml 

fashion_data = tf.keras.datasets.fashion_mnist

(train_img, train_labels), (test_img, test_labels) = fashion_data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_img.shape
print(len(train_labels))


#preprocess 
plt.figure()
plt.imshow(train_img[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_img = train_img / 255.0
test_img = test_img /255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_img[i])
    plt.xlabel(class_names[train_labels[i]])
plt.show()



