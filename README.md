# Triplet Loss based Architecture for Anomaly Detection

## Introduction 

## Data Loading and Preprocessing 
The first step in our pipeline is data loading and preprocessing. The images are labelled as OK and NG, where 'NG' represent the anomaly(defected) images. Image info which is image path and label is stored in 'image_path.csv'. The images in this csv file represent the original images. After storing image info,the next part is to extract a rectangule around the circular cross section (area of concern). Below are the examples for the same. 

<p align="center">
  <img src="images/example.png?raw=true" alt="Sublime's custom image"/>
</p>

<p align="center"> Drawn rectangular box is extraced from the image and then used in the training process. </p>

#### Example of cropped images 

<p float="left">
  <img src="images/37-s-OK-0.bmp" width="200" height="200" />
  <img src="images/1-NG-8.bmp" width="200" height="200"/> 
  <img src="images/1-OK-11.bmp" width="200" height="200"/>
</p>
For drawing recatangular boxes around circular cross section, we are using Hough Circle Algorithm

We store the image path of cropped images in 'crop_image_paths.csv'. On top of of that each cropped image is subjected to a set of augmentations. These are:

* Rotation - (0, 90, 180, 270)
* Flip - (None, Horizontal, Vertical)

Using combinations of these augmentations we get 11 augmented images from one original image. This heavy augmentaion is required because we have a very limited dataset for our intial experiment. 

## Deep Learning Architecture
<p align="center">
  <img src="images/model_arch.png?raw=true" alt="Sublime's custom image"/>
</p>
