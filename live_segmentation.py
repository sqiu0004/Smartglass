# Description: tests the live segmentation given a camera input

import cv2
import numpy as np
import torch
import torchvision
import time
from PIL import Image
from torchvision.models.segmentation import segmentation


# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen

def load_model():
    MODEL_NAME = "./Models/bdd_city_320_w.pt"
    check_point = torch.load(MODEL_NAME)
    net.load_state_dict(check_point['model_state_dict'])

GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')

# loading model
net = segmentation.lraspp_mobilenet_v3_large(pretrained=False, progress=True, num_classes=3).to(device)
load_model()

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

                # Processing the image to the pre-trained model
                time2 = time.time()
                out = net(cam_img_t.to(device).float())['out']
                om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
                time3 = time.time()
                print(time3-time2)

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
