# Created by: Sheng Wen Senman Qiu and Minsoo Kim
# Description: recaives camera input, infers segmentation model, outputs segmented image onto OLEDS

#--------------Driver Library-----------------#
import Jetson.GPIO as GPIO
import OLED_Driver as OLED
import OLED_Driver2 as OLED2
#--------------Image Library------------------#
import cv2
from PIL  import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageColor
#--------------Other--------------------------#
import numpy as np
import torch
import torchvision
import traceback
from PIL import Image
from torchvision.models.segmentation import segmentation
import time

#--------------Model initialisation--------------------------#
def load_model():
    MODEL_NAME = "./Models/bdd_city_320_w.pt"
    check_point = torch.load(MODEL_NAME)
    net.load_state_dict(check_point['model_state_dict'])

GPU_indx = 0
device = torch.device(GPU_indx if torch.cuda.is_available() else 'cpu')

#--------------Camera options--------------------------#
def gstreamer_pipeline(
    capture_width=1640,
    capture_height=1232,
    display_width=320,
    display_height=320,
    framerate=30,
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

#-------------Run model---------------#
def Run_Model(cam_img):
    # Preprocess the camera feed
    cam_img_t=np.moveaxis(cam_img,-1,0)
    cam_img_t=torch.tensor(cam_img_t).unsqueeze(0)

    # Processing the image to the pre-trained model
#    time2 = time.time()
    out = net(cam_img_t.to(device).float())['out']
#    time3 = time.time()
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

    # Plotting the image overlay
    segmented_img = cam_img
    segmented_img[:,:,0] = np.where(om==1, np.ceil(cam_img[:,:,0]/255)*255, segmented_img[:,:,0])
    segmented_img[:,:,1] = np.where(om==2, np.ceil(cam_img[:,:,1]/255)*255, segmented_img[:,:,1])
    
#    om_nt = (cv2.bitwise_not(om))/255
#    print(np.unique(om))
#    print(np.unique(cv2.bitwise_not(om)))
#    final = np.multiply(cam_img,np.moveaxis([om_nt,om_nt,om_nt],0,-1))+segmented_img
    
    final = segmented_img
    return final

#-------------Display Functions---------------#
# converts live image to PIL and sends it to one of two OLEDs
def Display_Live(img,num):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    if num == 1:
        OLED.Display_Image(im_pil)
    else:
        OLED2.Display_Image(im_pil)
# converts static image to PIL and sends it to OLED 1
def Display_Static(File_Name):
    image = Image.open(File_Name)
    OLED.Display_Image(image)
# outputs are calibrated to user POV
def cutout(img, resolution):
    dim = 130
    bl1 = [90,400]
    bl2 = [270,400]
    img1 = img[bl1[1]-dim:bl1[1],bl1[0]:bl1[0]+dim]
    img2 = img[bl2[1]-dim:bl2[1],bl2[0]:bl2[0]+dim]
    img1 = cv2.resize(img1, (resolution,resolution), interpolation = cv2.INTER_AREA)
    img2 = cv2.resize(img2, (resolution,resolution), interpolation = cv2.INTER_AREA)
    return img1, img2

#----------------------MAIN-------------------------#
# loading model
net = segmentation.lraspp_mobilenet_v3_large(pretrained=False, progress=True, num_classes=3).to(device)
load_model()

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=4), cv2.CAP_GSTREAMER)

def main():
    #-------------OLED Init------------#
    OLED.Device_Init()
    OLED2.Device_Init()

    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            current_time1 = time.time()
            ret, img = cap.read() # camera input
            
            if ret:
                current_time2 = time.time()
                img = cv2.resize(img, (320,320), interpolation = cv2.INTER_AREA) # scales image for the model input
                img = Run_Model(img)
                img = cv2.resize(img, (640,480), interpolation = cv2.INTER_AREA) # rescales image for the real world
                cv2.imshow("CSI Camera", img)

                # -------------Draw Pictures------------#
                img1, img2 = cutout(img, 64)
                current_time3 = time.time()
                Display_Live(img1,1)
                Display_Live(img2,2)
                current_time4 = time.time()
                    
#                print("Capture stage: ", current_time2-current_time1)
    #            print("Model stage: ", current_time3-current_time2)
                print("Display stage: ", current_time4-current_time3)

                # This also acts as
                keyCode = cv2.waitKey(1) & 0xFF
                # Stop the program on the ESC key
                if keyCode == ord('q'):
                    break
                    
        print("\r\nExit")
        cap.release()
        cv2.destroyAllWindows()
        OLED.Clear_Screen()
        OLED2.Clear_Screen()
        GPIO.cleanup()       
           
    else:
        print("Unable to open camera")

if __name__ == '__main__':
    try:
        main()
    except:
        if Exception:
            traceback.print_exc()
        print("\r\nError")
        cap.release()
        cv2.destroyAllWindows()
        OLED.Clear_Screen()
        OLED2.Clear_Screen()
        GPIO.cleanup()  


