# MIT License
# Copyright (c) 2019 JetsonHacks
# See license
# Using a CSI camera (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit using OpenCV
# Drivers for the camera and OpenCV are included in the base image

import cv2
import numpy as np
import torch
import torchvision
import time
#import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models.segmentation import segmentation

# Libraries added
import tensorflow as tf


# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./Models/LRASPP.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def gstreamer_pipeline(
    capture_width=1640,
    capture_height=1232,
    display_width=320,
    display_height=320,
    framerate=29,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera():
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=4), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)

        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, cam_img = cap.read()
            
            if ret_val:    
                # Preprocess the camera feed
                cam_img_t=np.moveaxis(cam_img,-1,0)
                cam_img_t=torch.tensor(cam_img_t).unsqueeze(0)
                cam_img_t = np.array(cam_img_t)
                cam_img_t = cam_img_t.astype('float32')
                t1 = time.time()

                #########################################################
                # Parts which were added

                # Image input shape needs to be (1, 3, 320, 320)
                # input_data = processed_img_t.numpy()
                interpreter.set_tensor(input_details[0]['index'], cam_img_t)
                interpreter.invoke()

                # The function `get_tensor()` returns a copy of the tensor data.
                # Use `tensor()` in order to get a pointer to the tensor.
                output_data = interpreter.get_tensor(output_details[0]['index'])
                om = np.argmax(output_data.squeeze(), axis=0)
                #########################################################
                t2 = time.time()
                print(t2-t1)





                # Processing the image to the pre-trained model
            #    time2 = time.time()
            #    out = net(cam_img_t.to(device).float())['out']
            #    time3 = time.time()
            #    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

                # Plotting the image overlay
                segmented_img = cam_img*0
                segmented_img[:,:,0] = np.where(om==1, cam_img[:,:,0], segmented_img[:,:,2])
                segmented_img[:,:,1] = np.where(om==2, cam_img[:,:,1], segmented_img[:,:,2])

                cv2.imshow("CSI Camera", segmented_img)

            keyCode = cv2.waitKey(1) & 0xFF
            
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":

    show_camera()
