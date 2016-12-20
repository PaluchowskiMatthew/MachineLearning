# Project 2 - Road Segmentation

Marie Drieghe, Lazare Girardin, Mateusz Paluchowski @ EPFL 2016

### Table of Contents  
[Abstract](#Abstract)    
[Data](#Data)    
[Feasibility & Constraints](#Feasibility)     
[Deliverables](#Deliverables)      
[Timeplan](#Timeplan)  
[Libraries and Frameworks](#Frameworks)    

<a name="Abstract"/>
## Abstract
For this choice of project task, we are provided with a set of satellite images acquired from GoogleMaps and also ground-truth images where each pixel is labeled as road or background.
The task is to train a classifier to segment roads in these images, i.e. assigns a label road=1, background=0 to each pixel.

<a name="Data"/>
## Data
The image datasets are available from the course kaggle page here:
[https://inclass.kaggle.com/c/epfml-segmentation]

<a name="Feasibility"/>
## Feasibility & Constrains
You are allowed to use any external library and ML techniques, as long as you properly cite any external code
used.

Obtain the python notebook segment_aerial_images.ipynb from this github folder, to see example code on how to extract the images as well as corresponding labels of each pixel.
Example code here also provides helper functions to visualize the images, labels and predictions. In particular, the two functions mask_to_submission.py and submission_to_mask.py help you to convert from the submission format to a visualization, and vice versa.

<a name="Deliverables"/>
## Deliverables
### Kaggle
  The better the score, the higher the grade!
### Report 
  Your 4 page report as .pdf
### Code
  The complete executable and documented Python code, as one .zip file.
Rules for the code part:
- Reproducibility: In your submission, you must provide a script run.py which produces exactly the same .csv predictions which you used in your best submission to the competition on Kaggle. This includes a clear ReadMe file explaining how to reproduce your setup, including precise training, prediction and installation instructions if additional libraries are used - the same way as if you would ideally instruct a new teammate to start working with your code.
- Documentation: Your ML system must be clearly described in your PDF report and also well documented in the code itself. A clear ReadMe file must be provided. The documentation must also include all data preparation, feature generation as well as cross-validation steps that you have used.
- External ML libraries are allowed, as long as accurately cited and documented.
- External datasets are allowed, as long as accurately cited and documented.

<a name="Timeplan"/>
## Timeplan
- *Nov 29, 2016* Do a reaserch on state-of-the-art methods and approaches. Highlight most important findiings.
- Performance measurments agains baseline model (Sketch jupyter notebook by *Nov 24, 2016*)
- Feature engineering
- Choosing the most promising method 
- **Deadline** Dec 22, 2016

<a name="Frameworks"/>
## Libraries and Frameworks
- NumPy
- Keras
- [tqdm](https://github.com/noamraph/tqdm) - for-loop progress bar 

## File structure
- Helpers: python files containing helpers for pre-processing, post-processing, perfomance measurements, visualization and submission
- Models: folder containing all trained models mentioned in the report
- Baseline.ipynb: implementation of a random baseline
- KNeighborsClassifier.ipynb: implementation of a K Nearest Neighbors approach
