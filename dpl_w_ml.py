import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import matplotlib as mpl
import mlflow as ml
from tensorflow import keras 

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


## actual model, very simple one, 5 layers, flattern, 4 dense, 128,64,32,10

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10)
    ])

##compile stage, loss function = , optimser=adam, metrics = 

model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
        )

## feed the model

model.fit(train_img, train_labels, epochs=20)

#evaluate it
test_loss, test_Acc = model.evaluate(test_img, test_labels, verbose=2)
print('\nTest accuracy:', test_Acc)

#Make predictions
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predict = probability_model.predict(test_img)
# print all the scores for the first test images
print(predict[0])
#print the one it is most confident about
print('\nWe believe it is: %s, and the right answer is %s.'% 
        (np.argmax(predict[0]), test_labels[0]))

