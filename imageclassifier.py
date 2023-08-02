import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np

data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

#os.listdir(os.path.join(data_dir, 'cats')) #all the cats images
os.listdir(data_dir)
for image_class in os.listdir(data_dir):
  for image in os.listdir(os.path.join(data_dir, image_class)):
    image_path = os.path.join(data_dir, image_class, image)
    try:
      img = cv2.imread(image_path)
      tip = imghdr.what(image_path)
      if tip not in image_exts:
        print(f"Image not in list {image_path}")
        os.remove(image_path)
    except Exception as e:
      print(f"Issue with image {image_path}")

data = tf.keras.utils.image_dataset_from_directory('data') #is not a preloaded dataset, is a generator, cant just call, need to make it a numpy iterator

data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

batch[0].shape #images as numpy arrays

f, axarr = plt.subplots(1)
print(axarr, f)
axarr = plt.imshow(batch[0][1].astype(int))
print(batch[1][1])

scaled = batch[0] / 255

scaled.max()

"""# Preprocess Data"""

data = data.map(lambda x,y: (x/255, y))

scaled_iterator = data.as_numpy_iterator()

batch = scaled_iterator.next()

len(data)

train_size = int(len(data)*0.6)
val_size = int(len(data)*0.2)
test_size = int(len(data)*0.2)
print(train_size, val_size, test_size)

train = data.take(train_size) #how much it takes from the batch
val = data.skip(train_size).take(val_size) #skips the ones taken by the train and takes the val size
test = data.skip(train_size+val_size).take(test_size) #same as above

"""# Deep Model"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential() #layers can go in here

model.add(Conv2D(16, (3,3), 1, activation="relu", input_shape=(256, 256, 3))) #16 filters (take relevant information), 3 x 3 pixels in size, stride of 1, move 1 pixel at a time
model.add(MaxPooling2D()) #condense information max value in a 2 x 2 region

model.add(Conv2D(32, (3,3), 1, activation="relu"))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation="relu"))
model.add(MaxPooling2D())

model.add(Flatten()) #flatten data to pass to dense, we dont want the channels makes it one dimensional

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
 #optimizer, loss(binary for two cases), metrics

model.summary() #see how model transforms data

"""# Train"""

logdir='logs'

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
plt.plot(hist.history['accuracy'], color='blue', label='accuracy')
plt.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

"""#Evaluate Performance"""

from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
ba = BinaryAccuracy()

for batch in test.as_numpy_iterator():
  x, y = batch
  yhat = model.predict(x)
  pre.update_state(y,yhat)
  re.update_state(y,yhat)
  ba.update_state(y,yhat)

print(f'Precision: {pre.result()},Recall: {re.result()},BinaryAccuracy: {ba.result()}')

img = cv2.imread('dogtest.jpg')
plt.imshow(img)
plt.show()

resize = tf.image.resize(img, (256,256)) #model only takes images with 256 by 256 pixels
plt.imshow(resize.numpy().astype(int))
plt.show()

resize.shape

np.expand_dims(resize, 0).shape #model expect a batch of images, not just one. so we encapsulates the image in another list

yhat = model.predict(np.expand_dims(resize/255, 0)) #also scaling it
yhat

if yhat < 0.5:
  print("Its a Cat")
else:
  print("Its a Dog")

"""#Save Model"""

from tensorflow.keras.models import load_model

model.save(os.path.join("models","DogCatClassifier.h5" ))

new_model = load_model(os.path.join("models", "DogCatClassifier.h5"))

new_model.predict(np.expand_dims(resize/255, 0))