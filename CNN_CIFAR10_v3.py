#tsun 20220105
#CNN exercise on MNIST
#reference:
#https://www.tensorflow.org/tutorials/images/cnn
#https://www.kaggle.com/ektasharma/simple-cifar10-cnn-keras-code-with-88-accuracy


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


#------data prep------\
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

class_names = ['plane', 'auto', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


#------model components------\
#https://stackoverflow.com/questions/46841362/where-dropout-should-be-inserted-fully-connected-layer-convolutional-layer
model = models.Sequential()
model.add(layers.experimental.preprocessing.RandomFlip())

model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Dropout(0.4))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))    # num_classes = 10

model.compile(optimizer='adamax',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=100, 
                    validation_data=(test_images, test_labels))

model.summary()

#------training visualizing------\
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('test_acc',test_acc)


#------result review------\
test_pred = model.predict(test_images)
test_pred_show = test_pred.argmax(axis=1)
test_pred_boo = pd.Series((test_pred_show == test_labels.reshape(-1)))
class_names_df = pd.DataFrame(class_names).reset_index().rename(columns={'index':'label_num',0:'label_str'})

#print a few pred vs test, first 25 wrongs
test_pred_boo_wrong_index = test_pred_boo[test_pred_boo==False].index
test_images_wrong = test_images[test_pred_boo_wrong_index]

test_pred_wrong_wrong = pd.DataFrame(test_pred_show[test_pred_boo_wrong_index]).rename(columns={0:'label_num'})
test_pred_wrong_wrong = test_pred_wrong_wrong.merge(class_names_df,on='label_num',how='left')

test_pred_wrong_actual = pd.DataFrame(test_labels.reshape(-1)[test_pred_boo_wrong_index]).rename(columns={0:'label_num'})
test_pred_wrong_actual = test_pred_wrong_actual.merge(class_names_df,on='label_num',how='left')

test_pred_wrong_vs_actual = test_pred_wrong_wrong.label_str.str.cat(test_pred_wrong_actual.label_str,sep=' pred vs ')

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
#print a few pred vs test, first 25 wrongs
test_pred_boo_correct_index = test_pred_boo[test_pred_boo==True].index
test_images_correct = test_images[test_pred_boo_correct_index]

test_pred_correct_correct = pd.DataFrame(test_pred_show[test_pred_boo_correct_index]).rename(columns={0:'label_num'})
test_pred_correct_correct = test_pred_correct_correct.merge(class_names_df,on='label_num',how='left')

test_pred_correct_actual = pd.DataFrame(test_labels.reshape(-1)[test_pred_boo_correct_index]).rename(columns={0:'label_num'})
test_pred_correct_actual = test_pred_correct_actual.merge(class_names_df,on='label_num',how='left')

test_pred_correct_vs_actual = test_pred_correct_correct.label_str.str.cat(test_pred_correct_actual.label_str,sep=' pred vs ')

plt.figure(figsize=(10,8))
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images_correct[i])
    plt.xlabel(test_pred_correct_vs_actual[i])
plt.show()


