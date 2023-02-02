import tensorflow.keras.models as tkm
from utils import *
import pandas as pd

model_path ="C:\\Users\\nampo\\Pictures\\VOC_Dataset\\unet.h5"
data_path = "C:\\Users\\nampo\\Pictures\\VOC_Dataset\\"

model = tkm.load_model(model_path)

#Retrieve test dataset
X_test = pd.read_csv(os.path.join(data_path,"X_test"))
X_test = X_test.to_numpy()

y_test = pd.read_csv(os.path.join(data_path,"y_test"))
y_test = y_test.to_numpy()


input_size=(256, 256, 3)
nb_classes=21
nb_images = 299

#Reshape test dataset
dataset_test = np.reshape(X_test, (nb_images,input_size[0], input_size[1], input_size[2]))
labels_test = np.reshape(y_test, (nb_images,input_size[0], input_size[1], nb_classes))

#Prediction
y_predict = model.predict(dataset_test)

for i in range(y_predict.shape[0]):
    image = y_predict[i,:,:,:]
    seg_labels = tf.math.softmax(seg_labels, axis=-1)
    seg_labels = tf.argmax(seg_labels, axis=-1)
    seg_labels = np.array(seg_labels)
    output = decode_segmap(seg_labels)

    ground_truth = labels_test[i,:,:,:]
    ground_truth = tf.math.softmax(ground_truth, axis=-1)
    ground_truth = tf.argmax(ground_truth, axis=-1)
    ground_truth = np.array(ground_truth)
    ground_truth = decode_segmap(ground_truth)


    fig, axs = plt.subplots(3,1)
    axs[0].imshow(X_test[i,:,:,:])
    axs[1].imshow(ground_truth)
    axs[2].imshow(output)
    axs[0].text(5, 5, 'Original image', bbox={'facecolor': 'white', 'pad': 10})
    axs[1].text(5, 5, 'Ground truth masks', bbox={'facecolor': 'white', 'pad': 10})
    axs[2].text(5, 5, 'Output masks', bbox={'facecolor': 'white', 'pad': 10})
    plt.show()