#%% importing libraries
from keras.models import Sequential # importining the type of model
from keras.layers.core import Dense, Dropout, Activation, Flatten # import layers
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image
import numpy
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix



#%% Data

#path1 = "C:\\Users\\Varun\\Pictures\\cnn_images"
#path2 = "C:\\Users\\Varun\\Pictures\\cnn_image_resize"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


thelist=open(r'C:\Users\victo\Desktop\data elective\images_segm_zurich\clusters_out.txt','r').readlines()  #previous version using clusters obtained from segmentation





print(len(thelist),len([s for s in thelist if s.split('\n')[0].split(";")[1]=='1' ]),len([s for s in thelist if s.split('\n')[0].split(";")[1]=='0' ]),len([s for s in thelist if s.split('\n')[0].split(";")[1]=='2' ]),len([s for s in thelist if s.split('\n')[0].split(";")[1]=='-1' ]))

thelist0=[line  for line in thelist if line.split('\n')[0].split(";")[1]=='0']
thelist1=[line  for line in thelist if line.split('\n')[0].split(";")[1]=='1']

thelistm1=[line  for line in thelist if line.split('\n')[0].split(";")[1]=='-1']

#thelistm1=[]

#thelist=thelist0[0:min([len(thelist0),len(thelist1),len(thelistm1)])]+thelist1[0:min([len(thelist0),len(thelist1),len(thelistm1)])]+thelistm1[0:min([len(thelist0),len(thelist1),len(thelistm1)])] 
thelist=thelist0[0:min([len(thelist0),len(thelist1)])]+thelist1[0:min([len(thelist0),len(thelist1)])]+thelistm1

print(len(thelist),len([s for s in thelist if s.split('\n')[0].split(";")[1]=='1' ]),len([s for s in thelist if s.split('\n')[0].split(";")[1]=='0' ]),len([s for s in thelist if s.split('\n')[0].split(";")[1]=='2']),len([s for s in thelist if s.split('\n')[0].split(";")[1]=='-1' ]))


path1 = "C:\\Users\\victo\\Desktop\\data elective\\images_segm_zurich"
path2 = "C:\\Users\\victo\\Desktop\\data elective\\cnn_image_resize_zurich"

#listing = os.listdir(path1)
#num_smples=size(listing)
num_smples=len(thelist)
print("Num of samples: ",num_smples)

img_rows, img_cols = 200, 200

#for file in listing:
for file in [s.split('\n')[0].split(";")[0] for s in thelist]:
    im = Image.open(path1 + '\\' + file)
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L') # converting to gray for 1 channel
    
    gray.save(path2 + '\\' + file, "JPEG")
    
##imlist = os.listdir(path2)

imlist=[line.split('\n')[0].split(";")[0] for line in thelist ]

# Converting  input images into an array
im1 = np.asarray(Image.open("C:\\Users\\victo\\Desktop\\data elective\\cnn_image_resize_zurich"+ '\\' + imlist[0]))
m,n = im1.shape[0:2] # to get the size of the image
imnbr = len(imlist)

# create matrix to store all flatetend image
immatrix = np.asarray([np.asarray(Image.open("C:\\Users\\victo\\Desktop\\data elective\\cnn_image_resize_zurich"+ '\\' + im2)).flatten()
                for im2 in imlist], 'f')

#label = np.ones((num_smples,),dtype = int)
#label[0:5] = 0
#label[5:10] = 1
#label[10:15] = 2

label=np.asarray([s.split('\n')[0].split(";")[1] for s in thelist])    #previous version using clusters obtained from segmentation



#data,Lable = shuffle(immatrix,label, random_state=None)
data,Lable = shuffle(immatrix,label)
train_data = [data,Lable]



img = immatrix[8].reshape(img_rows,img_cols)
plt.imshow(img)
plt.imshow(img, cmap= 'gray')
print(train_data[0].shape)
print(train_data[1].shape)

#%% define variable for CNN

batch_size = 128
# number of outputs classes
#nb_classes = 4                                   ########################### WEIMAR HAS $ SUBJECTIVE RESPONSE TYPES 0 1 2 3 




nb_classes=len(set([s.split('\n')[0].split(";")[1] for s in thelist if s.split('\n')[0].split(";")[1]!='-1']))
print("n_classes without noise",nb_classes)

# number of epocs to train 10
nb_epoch = 10

#input image dimensions
img_rows, img_cols = 200, 200
# number of convolun filter to use
nb_filters = 32
# size of pooling area from max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#%% Data manupulation

(X,y) = (train_data[0],train_data[1])



#Spliting training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=4)



X_train = X_train.reshape(X_train.shape[0], img_cols, img_rows, 1)
X_test = X_test.reshape(X_test.shape[0], img_cols, img_rows, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255 # normalizing by deviding by heighst intensity
X_test /= 255

print('X_train shape:', X_train.shape)
print('training samples', X_train.shape[0]) 
print('testing samples', X_test.shape[0])


# convert class vectors to binary class metices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


i = 5 # visualizing the 4600th sample (just for example)

image_sample = numpy.zeros(shape=(img_rows,img_cols))
for x in range(0,img_rows):
    for y in range(0, img_cols):
       image_sample[x][y] = X_train[i,x,y,0]

plt.imshow(image_sample, interpolation='nearest')
print("labels : ", Y_train[i,:])


#%% Defining CNN model
model = Sequential()

# Adding convolution layer
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode = 'valid', input_shape=(img_cols, img_rows, 1)))

convout1 = Activation('relu') # Rectified linear unit
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv)) # Convlution layer
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool,nb_pool))) ##Pooling  layer
model.add(Dropout(0.5)) # Regularizing layer


model.add(Flatten()) # Opening convolution, i.e., converting the 2D/3D to 1D 
model.add(Dense(128)) # Hiden layer of neulral network with 128 neurones
model.add(Activation('relu'))
model.add(Dropout(0.5)) # Regulizer
model.add(Dense(nb_classes)) # Output layer 
model.add(Activation('softmax'))

#
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
# model with scaled gradien desent optimizer
#model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


#%% Training the models
model.fit(X_train, Y_train, 
          batch_size=batch_size, 
          nb_epoch=nb_epoch,
          verbose=1,
          validation_data=(X_test,Y_test))


model.fit(X_train, Y_train, 
          batch_size=batch_size, 
          nb_epoch=nb_epoch,
          verbose=1,
          validation_split=0.2)

#%% Evaluate

score = model.evaluate(X_test,Y_test, verbose=0)
print('Test score: ', score)
print('Test accuracy: ', score[1])
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])





y_pred = model.predict_classes(X_test)
print(y_pred)
print(y_test)
print(confusion_matrix(y_test, [str(y_) for y_ in y_pred]))


