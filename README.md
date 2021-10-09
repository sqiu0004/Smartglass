# Smartglass
ECE4094-4095 Smart Glasses project 2021

Pipeline: Camera input → Segmentation model → output to OLED → project to visor

Requirements:
For Deep learning training process to be completed on Google colab:
PyTorch
Numpy
Matplotlib
OpenCV

Files:
Pre-processing.ipynb: script used to extract pedestrians and vehicles from the segmentation dataset, and resize the image to be used for training the model.
Input: Directory of the image and labels
Output: Chosen resolution of image and labels in npz file

Train.ipynb: script used to apply data augmentation, train the model and save the model with the lowest loss output.
Input: Directory of the npz files
Output: Trained model

Live_test.ipynb: script used to load the saved model and examine the output of the model. It is also used to calculate the MIoU and the mean inference time of the model.
Input: Directory of the model, image input
Output: Segmentation mask of the image input

Torch2tflite.ipynb: script used to load a PyTorch model and convert it into Tensorflow Lite model.
Input: PyTorch model
Output: Tensorflow Lite model

Main.py: script used to infer the camera input the the model, and output the segmentation mask to the OLED display
Input: Camera image
Output: Segmentation Mask output

#-------------------Additional dependencies for running main code-------------------
Why? At the time of writing this, Torch and Torchvision cannot be downloaded conventionally. 
Install torch==1.9.1 & torchvision==0.10.1 by following:
https://qengineering.eu/install-pytorch-on-jetson-nano.html
