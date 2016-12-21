# Project 2 - Road Segmentation

Marie Drieghe, Lazare Girardin, Mateusz Paluchowski @ EPFL 2016

### Table of Contents  
[Abstract](#Abstract)    
[Data](#Data)    
[Feasibility & Constraints](#Feasibility)     
[Deliverables](#Deliverables)      
[Timeplan](#Timeplan)  
[Libraries and Frameworks](#Frameworks)   
[File Structure](#Structure)  
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
- NumPy
- Keras
- [tqdm](https://github.com/noamraph/tqdm) - for-loop progress bar 

<a name="Structure"/>
## File structure
- Helpers: python files containing helpers for pre-processing, post-processing, perfomance measurements, visualization and submission
- Models: folder containing all trained models mentioned in the report
- Baseline.ipynb: implementation of a random baseline
- KNeighborsClassifier.ipynb: implementation of a K Nearest Neighbors approach
- cnn.py: implementation of first neural network that performs road segmentation on patches of the provided images
- post_padded.py: implementation of post processing neural network that tries to clean up the predictions of the first neural network. It works on the predictions from the first CNN and tries to predict the center of a larger patch.
- predictions.py: the predict() function takes a primary neural net and a post processing neural net and uses them to make a prediction and outputs a submission file.
