# Project 2 - Road Segmentation

Marie Drieghe, Lazare Girardin, Mateusz Paluchowski @ EPFL 2016

### Table of Contents  
[Abstract](#Abstract)    
[Data](#Data)    
[Libraries and Frameworks](#Frameworks)   
[File Structure](#FileStructure)   
[Data Structure](#DataStructure)   
<a name="Abstract"/>
## Abstract
For this choice of project task, we are provided with a set of satellite images acquired from GoogleMaps and also ground-truth images where each pixel is labeled as road or background.
The task is to train a classifier to segment roads in these images, i.e. assigns a label road=1, background=0 to each pixel.

<a name="Data"/>
## Data
The image datasets are available from the course kaggle page here:
[https://inclass.kaggle.com/c/epfml-segmentation]

<a name="Frameworks"/>
## Libraries and Frameworks
- [NumPy](http://www.numpy.org/)
- [Scikit-learn](http://scikit-learn.org/)
- [Scikit-image](http://scikit-image.org/)
- [Pillow](https://pillow.readthedocs.io/en/3.4.x/)
- [Keras](https://keras.io/)
  - [TensorFlow](https://www.tensorflow.org/) as backend

<a name="FileStructure"/>
## File structure
- Helpers: python files containing helpers for pre-processing, post-processing, perfomance measurements, visualization and submission
- Models: folder containing all trained models mentioned in the report
- Baseline.ipynb: implementation of a random baseline
- KNeighborsClassifier.ipynb: implementation of a K Nearest Neighbors approach
- Multi Layer Perceptron.ipynb: implementation of a Multi Layer Perceptron approach
- cnn.py: implementation of first neural network that performs road segmentation on patches of the provided images. WARNING: This script was runned on AWS g2.2xlarge instance on GPU and it took significant amount of time. It is not recommended to run it on a laptop.
- post_cnn.py: implementation of post processing neural network that tries to clean up the predictions of the first neural network. It works on the predictions from the first CNN and tries to predict the center of a larger patch. WARNING: This script was runned on AWS g2.2xlarge instance on GPU and it took significant amount of time. It is not recommended to run it on a laptop.
- primary_CNN_model.h5: Primary CNN model file stored in [Hierarchical Data Format](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) (.h5) which was trained on whole dataset by running train_cnn() function from cnn.py file.
- secondary_CNN_model.h5: Secondary CNN model file stored in [Hierarchical Data Format](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) (.h5) which was trained on whole dataset by running post_padd_sec() function from post_cnn.py file.
- run.py: script file for creating our top Kaggle predictions based on the two aforementioned models. the predict() function takes a primary neural net and a post processing neural net and uses them to make a prediction and outputs a submission file.

File structure is as follows (alphabetical order):
- helpers
  - data_helpers.py
  - image_handling.py
  - submission_helper.py
- models
  - Baseline.ipynb
  - KNeighborsClassifier.ipynb
  - Multi Layer Perceptron.ipynb
  - POST
    - secondary_CNN_model.h5
  - primary_CNN_model.h5
- cnn.py
- run.py
- post_cnn.py
- test_set_images
  - test_1.png
  - ...
- training
  - groundtruth
    - satImage_001.png
    - ...
  - images
    - satImage_001.png
    - ...

<a name="DataStructure"/>
## Data structure
The file structure of train images dataset should remained unchanged, however we change the format of test dataset.
Thus input data should follow file structure as presented below:
- training
  - groundtruth
    - satImage_001.png
    - ...
  - images
    - satImage_001.png
    - ...
- test_set_images
  - test_1.png
  - ...
