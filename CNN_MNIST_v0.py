#tsun 20220105
#CNN exercise on MNIST
#reference:
#https://www.tensorflow.org/tutorials/images/cnn


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


#------data prep------\
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

#need to reshape greyscale MNIST
#(entry, height, width) to 
#(entry, height, width, channel) where channel = 1
train_shape = list(train_images.shape)
train_shape.append(1)
train_shape = tuple(train_shape)
train_images = np.reshape(train_images,train_shape)

test_shape = list(test_images.shape)
test_shape.append(1)
test_shape = tuple(test_shape)
test_images = np.reshape(test_images,test_shape)

#normalize each pixel to [0,1]
train_images, test_images = train_images / 255.0, test_images / 255.0

#seeing some samples
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
plt.show()


#------model components------\
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=train_images.shape[1:]))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adamax',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))


#------training visualizing------\
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)


#------result review------\
test_pred = model.predict(test_images)
test_pred_show = test_pred.argmax(axis=1)
test_pred_boo = pd.Series((test_pred_show == test_labels))

#print a few pred vs test, first 25 wrongs
test_pred_boo_wrong_index = test_pred_boo[test_pred_boo==False].index
test_images_wrong = test_images[test_pred_boo_wrong_index]
test_pred_wrong = pd.Series(test_pred_show[test_pred_boo_wrong_index]).astype(str)
test_pred_wrong_actual = pd.Series(test_labels[test_pred_boo_wrong_index]).astype(str)
test_pred_wrong_vs_actual = test_pred_wrong.str.cat(test_pred_wrong_actual,sep=' pred vs actual ')

plt.figure(figsize=(10,8))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images_wrong[i])
    plt.xlabel(test_pred_wrong_vs_actual[i])
plt.show()

#print a few pred vs test, first 25 corrects
test_pred_boo_correct_index = test_pred_boo[test_pred_boo==True].index
test_images_correct = test_images[test_pred_boo_correct_index]
test_pred_correct = pd.Series(test_pred_show[test_pred_boo_correct_index]).astype(str)
test_pred_correct_actual = pd.Series(test_labels[test_pred_boo_correct_index]).astype(str)
test_pred_correct_vs_actual = test_pred_correct.str.cat(test_pred_correct_actual,sep=' pred vs actual ')

plt.figure(figsize=(10,8))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images_correct[i])
    plt.xlabel(test_pred_correct_vs_actual[i])
plt.show()


#what are the True Positve/TN/FP/FN?

