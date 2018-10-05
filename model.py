# Libraries: Image Processing and visualization
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from collections import Counter

# Libraries: Neural networks tools

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, pooling, MaxPooling2D, Activation, Dropout, BatchNormalization
from keras.layers import Cropping2D
from keras.initializers import TruncatedNormal
from keras.layers.advanced_activations import ELU

# The Data 

lines = []
#with open('/Users/William/Workspace/AI/Self Driving Car/Behavioral Clonning/CarND-Behavioral-Cloning-P3-master/data_t/driving_log.csv') as csvfile:
with open('/Users/William/simulatorData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
print('Rows from the CSV \n', len(lines))


images = []
measurements = []

for counter, line in enumerate(lines,0):
    #camera = np.random.randint(3, size=1)
    
    # center images
    
    center_image_source_path = line[0]
    filename = center_image_source_path.split('/')[-1]
    current_path = '/Users/William/simulatorData/IMG/'+ filename
    center_image = plt.imread(current_path)
    images.append(center_image)
    measurement = float(line[3])
    measurements.append(measurement)
    
    # left image
    
    left_image_source_path = line[1]
    filename = left_image_source_path.split('/')[-1]
    current_path = '/Users/William/simulatorData/IMG/'+ filename
    left_image = plt.imread(current_path)
    images.append(left_image)
    measurement = float(line[3])
    measurements.append(measurement)
    
    
    # right image
    
    right_image_source_path = line[2]
    filename = left_image_source_path.split('/')[-1]
    current_path = '/Users/William/simulatorData/IMG/'+ filename
    right_image = plt.imread(current_path)
    images.append(right_image)
    measurement = float(line[3])
    measurements.append(measurement)
    

X_train = np.array(images)
y_train = np.array(measurements)

print('Training Data without augmentation: images \n',len(X_train) )
print('Training Data without augmentation:  Angles measurements \n', len(y_train))
print('one Line from the CSV File: ', line[2])



# Data augmentation

augmented_images, augmented_measurements = [], []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(np.fliplr(image))
    augmented_measurements.append(measurement*-1.0)
    
X_train_aug = np.array(augmented_images)
y_train_aug = np.array(augmented_measurements)


print('Training Data with augmentation: images \n', len(X_train_aug))
print('Training Data with augmentation:  measurements \n', len(y_train_aug))
plt.imshow(X_train_aug[-1])
print(y_train_aug[-100])


# Dataset exploring 

ocurrences_in_data_set = Counter(y_train_aug)
top_imbalancers = ocurrences_in_data_set.most_common(3)
print(top_imbalancers)



# Generator 

def generator(data, batch_size):
    num_samples = len(data[0])
    
    while 1:
        for offset in range (0, num_samples, batch_size):
            batch_path = data[0][offset:offset + batch_size]
            batch_angle = data[1][offset:offset + batch_size]
            batch_rnd_sel = data[2][offset:offset + batch_size]
            
            images, angles = [], []
            
            for path, angle, rnd_sel in zip(batch_path, batch_angle, batch_rnd_sel):
                img = mpimg.imread(path)
                if rnd_sel == 1:
                    images.append(np.fliplr(img))
                    angles.append(angle)
                    
                else:
                    images.append(img)
                    angles.append(angle)
            
            batch_x_data = np.array(images)
            batch_y_data = np.array(angles)
            yield shuffle(batch_x_data, batch_y_data)


# The model 

X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug)
model = Sequential()

#### The Layers ####
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping= ((70, 25), (0,0))))
model.add(Conv2D(24, (5, 5), strides=(2,2),  activation= 'relu'))
#model.add(BatchNormalization())
model.add(Conv2D(36, (5, 5), strides=(2,2), activation = 'relu'))
#model.add(BatchNormalization())
model.add(Conv2D(48, (5, 5), strides=(2,2), activation = 'relu'))
#model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), strides=(1,1), activation = 'relu'))
#model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), strides=(1,1), activation = 'relu'))

######## End of Convolutions ######################



#model.add(BatchNormalization())
model.add(Flatten())
model.add(BatchNormalization())


########### Fully Connected Layers ##################
model.add(Dense(100))
model.add(Dropout(0.5))
#model.add(BatchNormalization())
#model.add(Dropout(0.5))
model.add(Dense(50))
#model.add(Dropout(0.25))
#model.add(BatchNormalization())
model.add(Dense(10))
#model.add(Dropout(0.5))
#model.add(BatchNormalization())
model.add(Dense(1))

#### End: The Layers ####

# Compiling
model.compile(loss = 'mse', optimizer = 'adam')
# The Loss
model.fit_generator(train_generator, samples_per_epoch=len(train_data[0]), validation_data=valid_generator,nb_val_samples=len(valid_data[0]), nb_epoch=3)

model.summary()
# Visualization
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()



# Saving the Model
model.save('model.h5')