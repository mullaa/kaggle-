# kaggle- Severstal: Steel Defect Detection
## Product Definition
  Product Mission: The mission of this product is to be able to detect defects in steel during manufacturing with a machine learning algorithm. Steel is used in a variety of ways such as in buildings, railways, roads, and other infrastructures. Skyscrapers, airports, bridges, and stadiums are just some of the examples of buildings that occupy a large amount of people. Defects in steel used in the construction of these buildings could result in a catastrophic event if the steel were to fail. The goal of our project is divided to 2 parts: first, localizing the defects' areas and locations. Second, classifying the defects. 
  
The following are 4 types of steel defects our machine learning algorithm will be able to detect:

Class 1
<img src="https://github.com/mullaa/kaggle-/blob/master/pictures/defects/class%201.png">

Class 2
<img src="https://github.com/mullaa/kaggle-/blob/master/pictures/defects/class%202.png">

Class 3
<img src="https://github.com/mullaa/kaggle-/blob/master/pictures/defects/class%203.png">

Class 4
<img src="https://github.com/mullaa/kaggle-/blob/master/pictures/defects/class%204.png">

Multi-class

<img src="https://github.com/mullaa/kaggle-/blob/master/pictures/defects/2%20label.png">

Triple-class

<img src="https://github.com/mullaa/kaggle-/blob/master/pictures/defects/3%20label.png">
  
  Target User(s): Severstal and other steel manufacturers
  
  User Stories:
  
  The user wants to help make production of steel more efficient by identifying defect.
  
  The user will provide a set of images with potential defects from high frequency cameras to power a defect detection classifier algorithm.
  
  The user can get the location of the surface defects on a steel sheet.
  
  The user can classify surface defects on a steel sheet.
  
  The companies can use our algorithm to  improve the qualities of their products.
  
  ## MVP
  The product will at a minimum be able to classify the image, its defect pixel location and output results into a folder for the user to view. The product will identify with moderate accuracy all four class type defects. For this product, a user interface is not required, but if time permits, we will create a simple user interface to allow the user to point to the directory of images for classification. There will be another window that outputs all the image defects for viewing, along with their filename. Also, if times permits, we will refine our algorithm to be able to identify multi-label defects and triple-label defects.
 
  ## Initial Plan of Attack
  -Have each person in the team select a classification method and try to implement it on a small data set.
  
  -Report their findings and we all determine the pros and cons of each classification method.
  
  -We all agree on a dataset and start to really work on optimizing the classification.
  
  -After picking a classification method, we begin optimization on a small training set and test set classification. The reason why we try small data sets first is because images have many pixels. This in turn increases the processing time. 
  
  -Obtain the relevant permissions needed to access BU's supercomputer.
  
 ## Product Survey 
  Existing similar products:
  Several different users who have attempted the Kaggle competition
  Severstal currently has an algorithm but will be refining based on user attempts.
  
  Example of some anomaly detection papers
  https://www.researchgate.net/publication/224207917_Automatic_Detection_of_Surface_Defects_on_Rolled_Steel_Using_Computer_Vision_and_Artificial_Neural_Networks
  
  http://people.idsia.ch/~juergen/ijcnn2012steel.pdf
  
  http://mit.imt.si/Revija/izvodi/mit171/zhou.pdf
  
  Research papers google drive link https://drive.google.com/drive/u/2/folders/0AHsq8E1UVd5mUk9PVA

## System Design

The flowchart of the system:

<img src="https://github.com/mullaa/kaggle-/blob/master/pictures/defects/Kaggle_flowchart.png">
  
  Major Components you think you will use:
    
    OpenCV library
    
    Matplotlib
    
    Python
    
    Tensorflow
    
    Torch
    
    Keras

  Technology Selection and reason behind selection including comparisons:
    OpenCV library is widely used and supported.
    Python is one of the languages required by the Kaggle competition
    Different algorithm. Eg: Anomaly detection,k-nearest neighbors/k-NN, SVM, logistic regression, random forest, and CNN (Convolution Neural Networks)
    CNN apparently does all the busy work to extract the features from the images. It handles the entire feature engineering part. In typical CNN architectures, there are multiple layers. Beginning layers are extracting the low-level features and end level layers extract high-level features from the image.
    The difficulty for images is extracting the correct features for classification which apparently CNN solves.

Before CNN, we need to spend time on selecting the proper features for classifying the image.
    
    Any test or verification programs
    
  Testing scenario:
  
    Separate a small set of defect images for testing.
    Compared the predicted output csv with testing output csv and determine error detection rate.
    Visual test on a small sample first to verify the algorithm is working correctly. 
    Eg: start with 50 samples, use OpenCV library to plot predicted anomaly locations and refine algorithm. 
    Afterwards, try another set of 50 samples. In the end, try algorithm on larger data sets.


