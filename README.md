# kaggle-
## Product Definition
  Product Mission: to detect defects in steel with machine learning algorithm
  
  Target User(s): Severstal
  
  User Stories:
  The user wants to help make production of steel more efficient by identifying defect.
  The user will provide a set of images with potential defects from high frequency cameras to power a defect detection      algorithm.
  The user can get the location of the surface defects on a steel sheet.
  The user can classify surface defects on a steel sheet.
  The companies can use our algorithm to  improve the qualities of their products.
  
  MVP
  (Severstal)User Interface Design for main user story if required - for this product a user interface is not required.   However, the product will provide images with boxes clearly identifying defects in the steel.
  
 ## Product Survey 
  Existing similar products:
  Several different users who have attempted the Kaggle competition
  Severstal currently has an algorithm but will be refining based on user attempts.
  
  Patent Analysis:

## System Design
  Major Components you think you will use:
    OpenCV library
    Matplotlib
    Python

  Technology Selection and reason behind selection including comparisons:
    OpenCV library is widely used and supported.
    Python is one of the languages required by the Kaggle competition
    Different algorithm. Eg: Anomaly detection,k-nearest neighbors/k-NN, One-class SVM 
    Any test or verification programs
    Visual test on a small sample first to verify the algorithm is working correctly. Eg: start with 50 samples, use OpenCV       library to plot predicted anomaly locations and refine algorithm. Afterwards, try another set of 50 samples. In the end,       try algorithm on larger data sets.


