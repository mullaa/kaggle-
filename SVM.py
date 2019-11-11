import pandas as pd
import numpy as np
import colorlover as cl
import plotly.graph_objs as go
from fastai.vision import *
from sklearn import svm, metrics
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
import cv2
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)

# reading in the training set
data = pd.read_csv('./severstal-steel-defect-detection/train.csv')
#train_data = pd.read_csv('./severstal-steel-defect-detection/train.csv')
#print(train_data)

# isolating the file name and class
data['ImageId'], data['ClassId'] = data.ImageId_ClassId.str.split('_', n=1).str
#data['ClassId'] = data['ClassId'].astype(np.uint8)

# storing a list of images without defects for later use and testing
no_defects = data[data['EncodedPixels'].isna()] \
                [['ImageId']] \
                .drop_duplicates()

# adding the columns so we can append (a sample of) the dataset if need be, later
no_defects['EncodedPixels'] = ''
no_defects['ClassId'] = np.empty((len(no_defects), 0)).tolist()
no_defects['Distinct Defect Types'] = 0
no_defects.reset_index(inplace=True)
# keep only the images with labels
squashed = data.dropna(subset=['EncodedPixels'], axis='rows', inplace=True)

# squash multiple rows per image into a list
squashed = data[['ImageId', 'EncodedPixels', 'ClassId']] \
            .groupby('ImageId', as_index=False) \
            .agg(list) \

# count the amount of class labels per image
squashed['Distinct Defect Types'] = squashed.ClassId.apply(lambda x: len(x))

# display first ten to show new structure
print(squashed.head(10))

# see: https://plot.ly/ipython-notebooks/color-scales/
colors = cl.scales['4']['qual']['Set3']
labels = np.array(range(1,5))

# combining into a dictionary
palette = dict(zip(labels, np.array(cl.to_numeric(colors))))
# we want counts & frequency of the labels
classes = data.groupby(by='ClassId', as_index=False) \
    .agg({'ImageId': 'count'}) \
    .rename(columns={'ImageId': 'Count'})

classes['Frequency'] = round(classes.Count / classes.Count.sum() * 100, 2)
classes['Frequency'] = classes['Frequency'].astype(str) + '%'

# plotly for interactive graphs
fig = go.Figure(

    data=go.Bar(
        orientation='h',
        x=classes.Count,
        y=classes.ClassId,
        hovertext=classes.Frequency,
        text=classes.Count,
        textposition='auto',
        marker_color=colors),

    layout=go.Layout(
        title='Defect: Count & Frequency with different color',
        showlegend=False,
        xaxis=go.layout.XAxis(showticklabels=False),
        yaxis=go.layout.YAxis(autorange='reversed'),
        width=750, height=400
    )
)

# display
#fig.show()

# count the combinations of labels and show the frequency
combinations = pd.DataFrame(data=squashed.ClassId.astype(str).value_counts())
combinations['Frequency'] = round(combinations.ClassId / combinations.ClassId.sum() * 100, 2)
combinations['Frequency'] = combinations['Frequency'].astype(str) + '%'

# plotly for interactive graphs
fig = go.Figure(

    data=go.Bar(
        orientation='h',
        x=combinations.ClassId,
        y=combinations.index,
        hovertext=combinations.Frequency,
        text=combinations.ClassId,
        textposition='auto'),

    layout=go.Layout(
        title='Defect Combinations in Images',
        showlegend=False,
        xaxis=go.layout.XAxis(showticklabels=False),
        yaxis=go.layout.YAxis(autorange='reversed'),
        width=750, height=500
    )
)
# display
#fig.show()

train, validate = train_test_split(squashed, test_size=0.2, random_state=0)

# print(train.head(5))
# print(validate.head(5))
print(train['ClassId'].astype(str).value_counts(normalize=True))
print('')
print(validate['ClassId'].astype(str).value_counts(normalize=True))

#X = squashed['ImageId']
#y = squashed['ClassId']
#print(X)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234123)
#clf = svm.SVC(kernel='linear')
#clf.fit(X_train, y_train)

#prediction = clf.predict(X_test)
#print("accuracy:", metrics.accuracy_score(y_test, y_pred=prediction), "\n")
#print("Classification report for - \n{}:\n{}\n".format(
#    clf, metrics.classification_report(validate, prediction)))

# labels = []
#for i in range(len(data)):
#    if type(data.EncodedPixels[i]) == str:
#        labels.append(1)
#    else:
#        labels.append(0)
#labels = np.array(labels)
#labels = labels.reshape((int(len(data)/4),4))
#print(labels.shape)

#images_df = pd.DataFrame(data.iloc[::4,:].ImageId_ClassId.str[:-2].reset_index(drop=True))
#labels_df = pd.DataFrame(labels.astype(int))
#proc_train_df= pd.concat((images_df,labels_df),1)
#print(proc_train_df)

#print(data.head())
x = []
height = 256
weight = 1600
### load images
for i in range(len(data['ClassId'])):
    #filename = str(data.sample(i).ImageId_ClassId.values)[2:]
    #filename = filename[:-4]
    fn = data['ImageId'].iloc[i]
    #fn = squashed['ImageId_ClassId'].iloc[i].split('_')[0]
    img = cv2.imread('./severstal-steel-defect-detection/train_images/' + fn)
    #filename = "./severstal-steel-defect-detection/train_images/"+fn
    # print(filename)

    img = cv2.resize(img, (height, weight))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x += [img]

x  = np.array(x)
print(x)
#imgplot = plt.imshow(img)
#plt.show()

y = []
for i in range(len(data['ClassId'])):
    y += [data['ClassId']]
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 0, test_size = 0.2)
clf = svm.SVC(kernel="linear", C=0.025)
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
print("accuracy:", metrics.accuracy_score(y_test, y_pred=prediction), "\n")



###split train test val
#val = data[0:1000]
#test = data[1000:2000]
#train = data[2000:]

###convert images to HOG vector
#def my_extractHOG(filename):
#    filename = str(filename)
#    filename = filename[:-2]
#    filename = "./severstal-steel-defect-detection/train_images/" + filename
#    img = mpimg.imread(filename)
#    img = cv2.resize(img, dsize=(600, 70), interpolation=cv2.INTER_CUBIC)
#    print(str(i)+"/"+str(train.ImageId_ClassId.shape[0]))
#    img = img / 256
#    fd,hog_image = hog(img, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualize=True)
#    return fd,hog_image

#ppc = 16
#hog_images = []
#hog_features = []

#for i, filename in enumerate(train.ImageId_ClassId):
#    fd,hog_image = my_extractHOG(filename)
#    if i<6 : hog_images.append(hog_image) # save some of images for example purpose only
#    hog_features.append(fd)

# print(hog_features.head())
#sc_X = StandardScaler()
#data_set = data[0:5095]
#x = data_set['ClassId'].values.reshape(-1,1)
#clf = svm.SVC(C=10, tol=1e-3, probability = True)
#clf.fit(hog_features, x)

#accuracy = clf.score(hog_features, x)
#print(accuracy)
