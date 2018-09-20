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





len1=len(open("C:\\Users\\victo\\Desktop\\data elective\\images_segm_zurich\\labels_count.txt" ).readlines())




thelist=open("C:\\Users\\victo\\Desktop\\data elective\\joint_set\\clusters.txt",'r').readlines()  





  




path2 = "C:\\Users\\victo\\Desktop\\data elective\\cnn_image_resize_joint"

#listing = os.listdir(path1)
#num_smples=size(listing)
num_smples=len(thelist)
print("Num of samples: ",num_smples)

img_rows, img_cols = 200, 200

list_im=[]
list_cluster=[]
lis_noise=[]


for city,city_cap,thelist_slice in [['zurich','Zurich',thelist[:len1]],['weimar','Weimar',thelist[len1:]]]:


    path1 = "C:\\Users\\victo\\Desktop\\data elective\\images_segm_%s"  % (city)
    #for file in listing:
    for file, cluster in [[s.split('\n')[0].split(";")[0],s.split('\n')[0].split(";")[1]] for s in thelist_slice]:
        im = Image.open(path1 + '\\' + file)
        img = im.resize((img_rows,img_cols))
        gray = img.convert('L') # converting to gray for 1 channel
    
        gray.save(path2 + '\\' +city+"_"+ file, "JPEG")
        if cluster!='-1':
            list_im.extend([city+"_"+ file])
            list_cluster.extend(cluster)
        else:
            lis_noise.extend([city+"_"+ file])
    





#####################BALANCE


#print('cluster sizes',[[x,list_cluster.count(x)] for x in set(list_cluster)])

#min_len= min([list_cluster.count(x) for x in set(list_cluster)])


#images,clusters= shuffle(list_im,list_cluster)


#list_bal=[]
#cluster_bal=[]

#for i in list(set(list_cluster)):


#    list_bal+=[images[j] for j in range(len(images)) if clusters[j]==i][0:min_len]
#    cluster_bal+=[clusters[j]  for j in range(len(images)) if clusters[j]==i][0:min_len]


#print(list_bal)
##imlist = os.listdir(path2)

#list_bal,cluster_bal=shuffle(list_bal,cluster_bal)



####################IMBALANCE


print('cluster sizes',[[x,list_cluster.count(x)] for x in set(list_cluster)])

min_len= min([list_cluster.count(x) for x in set(list_cluster)])

min_len_2= sorted([list_cluster.count(x) for x in set(list_cluster)])[1]


images,clusters= shuffle(list_im,list_cluster)


list_bal=[]
cluster_bal=[]

for i in list(set(list_cluster)):


    list_bal+=[images[j] for j in range(len(images)) if clusters[j]==i][0:min_len_2]
    cluster_bal+=[clusters[j]  for j in range(len(images)) if clusters[j]==i][0:min_len_2]


print(list_bal)
#imlist = os.listdir(path2)

list_bal,cluster_bal=shuffle(list_bal,cluster_bal)


imlist=list_bal                          #######images without noise






# Converting  input images into an array
#im1 = np.asarray(Image.open(path2+ '\\' + imlist[0]))
#m,n = im1.shape[0:2] # to get the size of the image
#imnbr = len(imlist)

# create matrix to store all flatetend image
immatrix = np.asarray([np.asarray(Image.open(path2+ '\\' + im2)).flatten()   for im2 in imlist], 'f')

immatrix_noise=np.asarray([np.asarray(Image.open(path2+ '\\' + im2)).flatten()
                for im2 in lis_noise], 'f')

#label = np.ones((num_smples,),dtype = int)
#label[0:5] = 0
#label[5:10] = 1
#label[10:15] = 2

#label=np.asarray([s.split('\n')[0].split(";")[1] for s in thelist])    

label=np.asarray(cluster_bal)

#data,Lable = shuffle(immatrix,label, random_state=None)
data,label_ = shuffle(immatrix,label)
train_data = [data,label_]

img = immatrix[8].reshape(img_rows,img_cols)
plt.imshow(img)
plt.imshow(img, cmap= 'gray')
print(train_data[0].shape)
print(train_data[1].shape)

#%% define variable for CNN

batch_size = 128
# number of outputs classes
#nb_classes = 4                                   ########################### WEIMAR HAS $ SUBJECTIVE RESPONSE TYPES 0 1 2 3 



nb_classes=len(set(list_cluster))
print('num classes:',nb_classes)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=4)     #originally 0.2

#############################################################################################################################################################              NOISE
#noise_len=int(0.05*len(X_train)/0.95)
#print(noise_len, 'initial noise len')
#noise_len+=(3-noise_len%3)

#label_noise=shuffle(np.repeat([0,1,2], noise_len/3).flatten())     #asisgning 0 and 1 labels to noise. since the dataset is balanced the number of 0 and 1 should be equal
#print(noise_len, 'noise labels:',label_noise)

#print('num of noise samples:',len(label_noise))


#X_train=np.append(X_train,immatrix_noise[0:noise_len], axis=0)    #add noise to traindata
#print('proportion of noise samples:',len(label_noise)/len(X_train))

#y_train=np.append(y_train,label_noise, axis=0) 

#print(len(X_train),len(y_train))
#############################################################################################################################################################              NOISE

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


