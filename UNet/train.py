from unet import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import pandas as pd

print(tf.test.gpu_device_name())

#All paths
registration_path = "C:\\Users\\nampo\\Pictures\\VOC_Dataset"
data_path = "C:\\Users\\nampo\\Pictures\\VOC_Dataset\\train_val.csv"
labels_path = "C:\\Users\\nampo\\Pictures\\VOC_Dataset\\labels.csv"

# Retrieve dataset
dataset = pd.read_csv(data_path)
dataset = dataset.to_numpy()
labels = pd.read_csv(labels_path)
labels = labels.to_numpy()

#Parameters for processing dataset
input_size=(256, 256, 3)
nb_classes=21
nb_images = 999

#Preprocess dataset
dataset = np.reshape(dataset, (nb_images,input_size[0], input_size[1], input_size[2]))
labels = np.reshape(labels, (nb_images,input_size[0], input_size[1], nb_classes))
X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.3, random_state=42)

#Save test set for future computations
X_test = pd.DataFrame(X_test)
X_test.to_csv(os.path.join(registration_path,"X_test"), index=False)
y_test = pd.DataFrame(y_test)
y_test.to_csv(os.path.join(registration_path,"y_test"), index=False)


#Parameters for training
EPOCHS = 6

#Model building
MyModel = unet(input_size = input_size, n_classes = nb_classes)
MyModel.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

#Model training
model_history = MyModel.fit(X_train,y_train, epochs=EPOCHS, batch_size=1)

#Save model
MyModel.save((os.path.join(registration_path,'unet.h5')))

plt.plot(model_history.history["accuracy"])
