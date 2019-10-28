#!/usr/bin/python
# coding=utf-8
import pandas as pd
import numpy as np
import os,sys
import random
from sklearn import svm,metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 10000000)
pd.set_option('display.max_colwidth', 1000000000)
# load the dataset
train_data = pd.read_csv("./severstal-steel-defect-detection/train.csv")
sample_data = pd.read_csv("./severstal-steel-defect-detection/sample_submission.csv")
# print(train_data.head())
# deal with data
train_not_null = [train_data[train_data['EncodedPixels'].notnull()]] # extract the not null rows
# print(train_not_null)
#
# train_data['EncodedPixels'].fillna(-1, inplace=True) # fill the dataset
# train_data['ClassId'] = train_data['ImageId_ClassId'].apply(lambda x: int(x[-1:])) # list a column of classID
# train_data['ImageId'] = train_data['ImageId_ClassId'].apply(lambda x: x[:-6] +  '.jpg' ) #list a column of ImageName
# train_data['Defect'] =np.where(train_data['EncodedPixels']==-1, 0, 1) # np.where(condition,x,y) if(condition){x} else{y} 有数据为1， 无数据为0
# train_data['ClassId'] =np.where(train_data['EncodedPixels']==-1,  0,train_data['ClassId']) #有数据的把class标上，没有的标0
# print(train_data.head())
train_data['ImageId'], train_data['ClassId'] = train_data.ImageId_ClassId.str.split('_', n=1).str
# storing a list of images without defects for later use and testing
no_defects = train_data[train_data['EncodedPixels'].isna()][['ImageId']].drop_duplicates()
# adding the columns so we can append (a sample of) the dataset if need be, later
no_defects['EncodedPixels'] = ''
no_defects['ClassId'] = np.empty((len(no_defects), 0)).tolist()
no_defects['Distinct Defect Types'] = 0
no_defects.reset_index(inplace=True)

# keep only the images with labels
squashed = train_data.dropna(subset=['EncodedPixels'], axis='rows', inplace=True)
# squash multiple rows per image into a list
squashed = train_data[['ImageId', 'EncodedPixels', 'ClassId']].groupby('ImageId', as_index=False).agg(list)
# count the amount of class labels per image
squashed['Distinct Defect Types'] = squashed.ClassId.apply(lambda x: len(x))
print(squashed.head(10))

sc_X = StandardScaler()
class_length = len(train_data['ClassId'])
# x = sc_X.fit_transform(train_data['ClassId'].values.reshape(class_length,-1))
pixels_length = len(train_data['EncodedPixels'])
# x = sc_X.fit_transform(train_data['EncodedPixels'].values.reshape(pixels_length,-1))

array_x = train_data['EncodedPixels'].values.reshape(pixels_length,-1).tolist()
x = array_x[-1:]
array_y = train_data['ClassId'].values.reshape(class_length,-1).tolist()
y = array_x[-1:]


np.random.seed(0)
X_train, X_test, Y_train, Y_test = train_test_split(x, array_y, test_size=0.3, train_size=0.7, random_state=109)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, Y_train)

prediction = clf.predict(X_test)
print("accuracy:", metrics.accuracy_score(y_true=Y_test, y_pred=prediction), "\n")
# print("Classification report for - \n{}:\n{}\n".format(
#     clf, metrics.classification_report(Y_test, y_pred)))



