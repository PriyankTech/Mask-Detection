# Real-Time Mask Detection System

This project is a Real-Time Mask Detection System that classifies whether a person is wearing a mask correctly, wearing it improperly, or not wearing a mask at all using Convolutional Neural Networks (CNNs). It leverages computer vision techniques to ensure public safety and health.

## Files Overview

1. **main.py**  
   - This is the core file of the project. It contains the implementation of the CNN model for mask detection. 
   - It includes data preprocessing, model architecture, training, and evaluation logic.
   - It also classifies images in real-time to detect mask status.

2. **upload.py**  
   - This script handles the uploading and preprocessing of new images for classification.
   - It ensures that images are resized and normalized before being fed into the model for prediction.
   - It facilitates testing the model on custom images.

3. **setup.bat**  
   - A batch script to automate the setup process.
   - It installs all required dependencies and sets up the environment for running the project smoothly.
   - Simply run this file to get started with the project without any hassle.

## Author
- **Priyank Fichadiya**  
  Computer Engineer passionate about developing intelligent systems that enhance user experiences.
  visit https://priyankvision.vercel.app/ for more info

## How to Run
1. Run `setup.bat` to install necessary dependencies.
2. Execute `main.py` to train and evaluate the model.
3. Use `upload.py` to test custom images for mask detection.

This project is a step towards ensuring public health safety with AI-powered mask detection. Happy coding!
