from utils import *
import pandas as pd

registration_path = "C:\\Users\\nampo\\Pictures\\VOC_Dataset\\"

train_path = "C:\\Users\\nampo\\Downloads\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\JPEGImages"

def create_csv(path, registration_path, csv_name, mask = False, num_classes = 21):
    if mask== False:
        dataset = np.zeros((1,256,256,3))
        iter = 1
        for r, d, f in os.walk(path):
            for file in f:
                #Take the first 1000 images
                if iter < 1000:
                    image = Image.open(os.path.join(path,file)).convert('RGB')

                    #If image's dimensions are less than 256x256, skip this image
                    if image.size[0]< 256 or image.size[1]< 256:
                        continue
                    #Resize image to 256x256
                    crop_rectangle = (0, 0, 256, 256)
                    cropped_im = image.crop(crop_rectangle)
                    cropped_im = np.array(cropped_im)
                    dataset = np.append(dataset, cropped_im[np.newaxis,...],axis = 0)
                    iter += 1
                else:
                    break
        #Delete the first element which is just composed of zeros
        dataset= np.delete(dataset, 0,axis = 0)

        #Reshape the dataset array into 2D array
        dataset = np.reshape(dataset, (dataset.shape[0]*dataset.shape[1]*dataset.shape[2], dataset.shape[3]))

        #Save the dataset as a csv file
        df = pd.DataFrame(dataset)
        df.to_csv(os.path.join(registration_path, csv_name), index=False)

    else: #When we are processing masks
        dataset = np.zeros((1,256, 256, 21))
        iter = 1
        for r, d, f in os.walk(path):
            for file in f:
                if iter < 1000:
                    print(iter)
                    #image = getMask3D(os.path.join(path,file))
                    image = cv2.imread(os.path.join(path,file))
                    if image.shape[0]< 256 or image.shape[1]< 256:
                        continue
                    image = image[:256,:256]

                    #mask_rgb is an array of dimensions 256x256x3
                    mask_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    #mask is an array of dimensions 256x256x1
                    mask = encode_segmap(mask_rgb)

                    #seg_labels is an array of dimensions 256x256x21
                    #Where each of the channel is a binary mask
                    seg_labels = tf.keras.utils.to_categorical(y=mask, num_classes=21)

                    dataset = np.append(dataset, seg_labels[np.newaxis,...],axis = 0)
                    iter += 1
                else:
                    break

        dataset = np.delete(dataset, 0, axis=0)
        dataset = np.reshape(dataset, (dataset.shape[0] * dataset.shape[1] * dataset.shape[2], dataset.shape[3]))

        df = pd.DataFrame(dataset)
        df.to_csv(os.path.join(registration_path,csv_name), index=False)

registration_path = "C:\\Users\\nampo\\Pictures\\VOC_Dataset\\"

train_path = "C:\\Users\\nampo\\Downloads\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\JPEGImages"
label_path = "C:\\Users\\nampo\\Downloads\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\SegmentationClass"

create_csv(path=train_path, registration_path=registration_path, csv_name="train_val.csv")

create_csv(path=label_path, registration_path=registration_path, csv_name="labels.csv", mask=True)



