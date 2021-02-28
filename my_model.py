
### Image and steering angle extraction
import csv 
import cv2 
import numpy as np
from scipy import ndimage

lines = []
with open('./data_store_2/driving_log.csv') as csvfile:
    my_read = csv.reader(csvfile)
    for line in my_read:
        lines.append(line)

### Initializers for the paramters to be used
images=[]
steering_angles = []
angle_store = 0 # used to smoothen the steering angles

count = 0 # Count the number of images 
correction = 0.2 # angle correction for left or right image
clr = 0 

for line in lines:
    
    clr =0
    for in_path in line[0:3]:

        filename = in_path.split('/')[-1]
        image_path = './data_store_2/IMG/' + filename

        ## Image Augmentation, All the odd lines are flipped

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #         print(image.shape)
        # image = image[cropy_start:cropy_end,:,:] ## Cropping the unneccsary portions of the image
        steering_angle = float(line[3])
        
        ## Smoothing the steering vectors
        steering_angle = 0.3*steering_angle + 0.7*angle_store
        angle_store = steering_angle
    
        count += 1
        images.append(image)
        steering_angle = angle_store + 0.2 * steering_angle

        if clr == 1:
            steering_angles.append(steering_angle + correction)
        elif clr == 2:
            steering_angles.append(steering_angle - correction)
        else:
            steering_angles.append(steering_angle)
        if clr == 0:  ### flip images if the image is from center camera
            image = np.fliplr(image)
        
            steering_angle = -steering_angle
            count += 1

            images.append(image)
            steering_angles.append(steering_angle)
        clr +=1
    
## Varibles 
Input_shape = images[0].shape
print(np.shape(images))

# Keras requires the data to be in arrays# 
X_train = np.array(images)
Y_train = np.array(steering_angles)

##### Creating a neural network architture 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D,Cropping2D, Dropout
from keras.layers import Lambda

## Architecture based NVIDIA Self driving Car
my_model = Sequential()
my_model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=Input_shape)) 
my_model.add(Lambda(lambda x: (x / 255) - 0.5))
my_model.add(Conv2D(24, (5, 5), activation='elu', subsample=(2, 2)))
my_model.add(Conv2D(36, (5, 5), activation='elu', subsample=(2, 2)))
my_model.add(Conv2D(48, (5, 5), activation='elu', subsample=(2, 2)))
my_model.add(Conv2D(64, (3, 3), activation='elu'))
my_model.add(Conv2D(64, (3, 3), activation='elu'))
my_model.add(Dropout(0.4))
my_model.add(Flatten())
my_model.add(Dense(100,activation='elu'))
my_model.add(Dense(50,activation='elu'))
my_model.add(Dense(10,activation='elu'))
my_model.add(Dense(1))
my_model.summary()
my_model.compile(loss='mse',optimizer='adam')
my_model.fit(X_train,Y_train,validation_split=0.2,shuffle=True, batch_size=128, epochs=10)
my_model.save('new_network_3.h5')
print('Model trained and saved')

    