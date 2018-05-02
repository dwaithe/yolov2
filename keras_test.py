import keras
from keras.layers import convolutional, Dense, Activation,pooling
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.callbacks import Callback,EarlyStopping,ModelCheckpoint
from keras import backend as K
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from scipy import ndimage
import numpy as np
import copy
from random import Random
class TestCallback(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
    def on_train_begin(self, logs={}):
        self.losses = []

    
def dice_coef(y_true, y_pred):
    """This is the default U-NET loss function which I do not use."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def dice_coef_loss(y_true, y_pred):
    """"""
    return -dice_coef(y_true, y_pred)
def accuracy_custom(y_true, y_pred):
    y_true_f = K.sum(y_true)
    y_pred_f = K.sum(y_pred)
    return (K.abs(y_true_f -y_pred_f)/y_true_f)
    



def split_the_images(X_imgs, Y_imgs,in_hei,in_wid,mag):
    
    """
    Function that splits images up in to smaller pieces and provides a boundary buffer region. 
    If the sampled pixels are from edge of input image then these are mirrored.
    
    Keyword arguments:
    
    X_imgs -- Input images.
    Y_imgs -- Corresponding output images.
    in_hei -- Desired patch height not inc. boundary.
    in_wid -- Desired patch width not inc. boundary.
    mag    -- The boundary size in pixels.
    
    Returns:
    
    train  -- An array of images for training which have been normalised for mean and variance.
    gtdata -- The corresponding ground-truth density representation.
    images_per_image -- The number of calculated tiles from an input image.
    
    """
    train = []
    gtdata = []
    for x_img, y_img in zip(X_imgs,Y_imgs):
        
        f_hei, f_wid = x_img.shape
        images_per_image = 0
        rows = 0

        for rst in range(0,f_hei,in_hei):
            rows +=1
            cols = 0
            for cst in range(0,f_wid,in_wid):
                cols +=1
                top = bottom = left = right = 0
                ren = rst + in_hei + 2*mag
                cen = cst + in_wid + 2*mag

                rst1 = rst
                cst1 = cst

                if rst ==0:
                    ren -= mag
                    top = 16
                else:
                    rst1 -= mag
                    ren -= mag

                if cst==0:
                    cen -= mag
                    left = 16
                else:
                    cst1 -= mag
                    cen -= mag


                if cen > f_wid:
                    right = cen-f_wid+1
                    cen = -1
                else:
                    right = 0


                if ren > f_hei:
                    bottom = ren-f_hei+1
                    ren = -1
                else:
                    bottom =0





                temp = np.copy(x_img[rst1:ren,cst1:cen])
               
                if top >0 or bottom >0 or left>0 or right >0:
                    temp = cv2.copyMakeBorder(temp, top, bottom, left, right, borderType=2)
                

                train.append(temp.reshape(1,temp.shape[0],temp.shape[1]))
                images_per_image += 1

                temp2 = y_img[rst1:ren,cst1:cen]
                if top >0 or bottom >0 or left>0 or right >0:
                    temp2 = cv2.copyMakeBorder(temp2, top, bottom, left, right, borderType=2)
                gtdata.append(temp2.reshape(1,temp2.shape[0],temp2.shape[1]))
    return train, gtdata,images_per_image
def get_unet(img_rows,img_cols):
    """This sets up the U-NET network structure. The same as in 
    https://github.com/jocicmarko/ultrasound-nerve-segmentation
    except that I use different activation function (not sigmoid) in the
    last layer and also I use a different loss function (not dice_coef)."""
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation=None)(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss='mse', metrics=['accuracy',accuracy_custom])

    return model




file_path = 'dataset01/'
data_store = {}
data_store['input'] = []
data_store['gt'] = []
data_store['dense'] = []

num_of_train = 160
sigma = 2.0
for i in range(0,num_of_train):
    n = str(i+1).zfill(3)
    
    #Open intensity image.
    img = Image.open(file_path+n+'cell.png').getdata()
    hei, wid = img.size
    temp = np.array(img).reshape((hei,wid,3))[:,:,2].astype(np.float32)
    temp -= np.mean(temp)
    temp /= np.std(temp)
    data_store['input'].append(temp)
    
    #Open ground-truth image.
    img =  Image.open(file_path+n+'dots.png').getdata()
    hei, wid = img.size
    data_store['gt'].append(np.array(img).reshape((hei,wid,3))[:,:,0].astype(np.float64))
    
    #Filter ground-truth image to produce density kernel representation
    data_store['dense'].append(ndimage.filters.gaussian_filter(data_store['gt'][i],sigma,mode='constant'))


train = []
gtdata = []
in_hei = 48 #96,112
in_wid = 48
mag = 16

X_trainf, X_testf, Y_trainf, Y_testf = train_test_split(data_store['input'],  data_store['dense'], train_size =8,test_size=100)
train_cut, train_gtdata_cut,images_per_image = split_the_images(X_trainf, Y_trainf,in_hei,in_wid,mag)
test_cut, test_gtdata_cut,images_per_image = split_the_images(X_testf, Y_testf,in_hei,in_wid,mag)
          

X_test = np.array(test_cut)
Y_test = np.array(test_gtdata_cut)



#Augment the data. I couldn't get what I wanted from Keras so I made my own.


temp_list_trainX = []
for tc in train_cut:
    temp_list_trainX.append(tc)
    temp_list_trainX.append(np.flipud(tc[0,:,:]).reshape(tc.shape))
    temp_list_trainX.append(np.fliplr(tc[0,:,:]).reshape(tc.shape))
    temp_list_trainX.append(np.rot90(tc[0,:,:]).reshape(tc.shape)) 
    temp_list_trainX.append(np.rot90(np.rot90(np.rot90(tc[0,:,:]))).reshape(tc.shape))
#Augment the data.
temp_list_trainY = []
for tc in train_gtdata_cut:
    temp_list_trainY.append(tc)
    temp_list_trainY.append(np.flipud(tc[0,:,:]).reshape(tc.shape))
    temp_list_trainY.append(np.fliplr(tc[0,:,:]).reshape(tc.shape))
    temp_list_trainY.append(np.rot90(tc[0,:,:]).reshape(tc.shape)) 
    temp_list_trainY.append(np.rot90(np.rot90(np.rot90(tc[0,:,:]))).reshape(tc.shape))


#subplot(1,4,1)
#imshow(temp_list_trainX[3][0,:,:])
#subplot(1,4,2)
#imshow(temp_list_trainY[3][0,:,:])


z = zip(temp_list_trainX, temp_list_trainY)
np.random.shuffle(z)
temp_list_trainX, temp_list_trainY = zip(*z)
print 'shuffling data'
#subplot(1,4,3)
#imshow(temp_list_trainX[3][0,:,:])
#subplot(1,4,4)
#imshow(temp_list_trainY[3][0,:,:])
    

X_train = np.array(temp_list_trainX)
Y_train = np.array(temp_list_trainY)


#Stores history of paramaters.
hist = keras.callbacks.History()


# combine generators into one which yields image and masks
#train_generator = zip(image_generator, mask_generator)
model = get_unet(X_train.shape[2],X_train.shape[3])
filepath="weights-improvement.hdf5"
print 'fitting model'
checkpoint = ModelCheckpoint(filepath,   verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]
model.fit(X_train, Y_train, batch_size=72, nb_epoch=1000,callbacks=callbacks_list,verbose=True)