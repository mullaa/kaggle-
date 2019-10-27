from glob import glob
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import cv2
import keras


batch_size = 32
num_classes = 4
epochs = 10

# input image dimensions
img_rows, img_cols = 256, 1600

# set paths to train and test image datasets
TRAIN_PATH = 'train_images/'
TEST_PATH = 'test_images/'

# load dataframe with train labels
train_df = pd.read_csv('train.csv')
train_fns = sorted(glob(TRAIN_PATH + '*.jpg'))
test_fns = sorted(glob(TEST_PATH + '*.jpg'))

print('There are {} images in the train set.'.format(len(train_fns)))
print('There are {} images in the test set.'.format(len(test_fns)))

train_df.head(10)

# split column
split_df = train_df["ImageId_ClassId"].str.split("_", n = 1, expand = True)

# add new columns to train_df
train_df['Image'] = split_df[0]
train_df['Label'] = split_df[1]

# check the result
train_df.head()

defect1 = train_df[train_df['Label'] == '1'].EncodedPixels.count()
defect2 = train_df[train_df['Label'] == '2'].EncodedPixels.count()
defect3 = train_df[train_df['Label'] == '3'].EncodedPixels.count()
defect4 = train_df[train_df['Label'] == '4'].EncodedPixels.count()
min_count = min([defect1, defect2, defect3, defect4])

path = TRAIN_PATH
count_one = 0;
count_two = 0;
count_three = 0;
count_four = 0;
count_one_test = 0;
count_two_test = 0;
count_three_test = 0;
count_four_test = 0;


x_train = []
y_train = []
x_test = []
y_test =[]

train_percent = .8
imgScale = .25

train_count = int(min_count*train_percent)
test_count = int((1-train_percent)*min_count)
images = sorted(glob(path + '*.jpg'))

for im in range(0, len(images)):
    label = int(train_df.iloc[im][3])-1

    orgimage = cv2.imread(path+train_df.iloc[im][2])
    newX,newY = orgimage.shape[1]*imgScale, orgimage.shape[0]*imgScale
    image = cv2.resize(orgimage,(int(newX),int(newY)))

    if(train_df.iloc[im][1]) != 'nan':
        if(label == 0):
            if(count_one <= train_count):
                count_one+=1
                x_train.append(image)
                y_train.append(label)
            elif(count_one_test <= test_count):
                count_one_test+=1
                x_test.append(image)
                y_test.append(label)
        if(label == 1):
            if(count_two <= train_count):
                count_two+=1
                x_train.append(image)
                y_train.append(label)
            elif(count_two_test <= test_count):
                count_two_test+=1
                x_test.append(image)
                y_test.append(label)
        if(label == 2):
            if(count_three <= train_count):
                count_three+=1
                x_train.append(image)
                y_train.append(label)
            elif(count_three_test <= test_count):
                count_three_test+=1
                x_test.append(image)
                y_test.append(label)
        if(label == 3):
            if(count_four <= train_count):
                count_four+=1
                x_train.append(image)
                y_train.append(label)
            elif(count_four_test <= test_count):
                count_four_test+=1
                x_test.append(image)
                y_test.append(label)

    print(len(y_test), len(x_test), len(y_train), len(x_train))
    print(count_four_test,count_three_test,count_two_test,count_one_test, test_count)
    all_data_count = test_count
    if(count_four_test >= all_data_count and
       count_three_test >= all_data_count and
       count_two_test >= all_data_count and
       count_one_test >= all_data_count):
        break;

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
y_train = keras.utils.to_categorical(y_train,4)
y_test = keras.utils.to_categorical(y_test,4)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu',
                 input_shape=(image.shape[0],image.shape[1],3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
