from unet import *
import matplotlib.pyplot as plt

#Parameters for training

model = unet()

unet.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

#Parameters for training
epochs = 5

#Training process
model_history = unet.fit(train_dataset, epochs=EPOCHS)

plt.plot(model_history.history["accuracy"])
