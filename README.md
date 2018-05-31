# Real-time hand gesture recognition
## Environment
python3, tensorflow, opencv2, theano, keras, Pillow, sklearn, matploat
## Usage
    python3 detect.py

Enter mode Train by typing 1 or use the pre-trained model by typing 2

Training model:

Start training the images in the folder named as img_folder

Else:

Some keyboard usage

    t - start tracking your hand
    
    d - filter your hand (need t)
    
    g - guess the gesture by some pre-trained model (need t and d)
    
    n - create a folder that will save training images for this time
    
    s - start saving current frame as training image (need t and d)