from glob import glob
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import cv2
import keras
import random
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.image as mpimg

num_classes = 4
train_percent = .8
imgScale = .5

# input image dimensions
img_rows, img_cols = 256, 1600

# set paths to train and test image datasets
TRAIN_PATH = '/media/sf_EC602/Kaggle_Steel/train_images/'
TEST_PATH = '/media/sf_EC602/Kaggle_Steel/test_images/'

# load dataframe with train labels
train_df = pd.read_csv('/media/sf_EC602/Kaggle_Steel/train.csv')
train_df = train_df.reindex(np.random.permutation(train_df.index))

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
print([defect1, defect2, defect3, defect4])
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

train_count = int(min_count*train_percent)
test_count = int((1-train_percent)*min_count)
images = sorted(glob(path + '*.jpg'))

for im in range(0, len(images)):
    label = int(train_df.iloc[im][3])-1

    orgimage = cv2.imread(path+train_df.iloc[im][2])
    newX,newY = (256,256)
    image = cv2.resize(orgimage,(int(newX),int(newY)))
    image = image.reshape(196608)
    # print(image.shape)

    # fd,hog_image = hog(image, orientations=8, pixels_per_cell=(16,16),cells_per_block=(4, 4),block_norm= 'L2',visualize=True)

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

    #print(len(y_test), len(x_test), len(y_train), len(x_train))
    #print(count_four_test,count_three_test,count_two_test,count_one_test, test_count)
    all_data_count = test_count
    if(count_four_test >= all_data_count and
       count_three_test >= all_data_count and
       count_two_test >= all_data_count and
       count_one_test >= all_data_count):
        break;

y_train_orig = y_train
y_test_orig = y_test
x_train = np.asarray(x_train)
# print(x_train.shape)
# print(x_train.type)
y_train = np.asarray(y_train)
# print(y_train.shape)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
# y_train = keras.utils.to_categorical(y_train,4)
# y_test = keras.utils.to_categorical(y_test,4)


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2,train_size = 0.8,random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
