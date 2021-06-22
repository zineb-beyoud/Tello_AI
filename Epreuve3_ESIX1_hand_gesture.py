import copy
import cv2
import threading
import socket
import sys
import time
import platform
from tensorflow.keras.models import load_model
import numpy as np
path="/Users/zinebb/Desktop/VGG_cross_validated.h5"
model = load_model(path) # open saved model/weights from .h5 file

### Drone settings
host = ''
port = 9000

locaddr = (host,port)
takeoff_cmd=1

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

tello1_address = ('192.168.10.1', 8889)
#tello2_address = ('192.168.77.102', 8889)
sock.bind(locaddr)

def recv():
    count = 0
    while True:
        try:
            data, server = sock.recvfrom(1518)
            print(data.decode(encoding="utf-8"))
        except Exception:
            print ('\nExit . . .\n')
            break
recvThread = threading.Thread(target=recv)
recvThread.start()

# Sélection mode SDK
sock.sendto("command".encode(encoding="utf-8"), tello1_address)
#sock.sendto("command".encode(encoding="utf-8"), tello2_address)
time.sleep(5)

###

# General Settings
prediction = ''
action = ''
score = 0
img_counter = 500

gesture_names = {
                 0: 'Fist',
                 1: 'L',
                 2: 'Okay',
                 3: 'Palm',
                 4: 'Peace'
}


def predict_rgb_image(img):
    result = gesture_names[model.predict_classes(img)[0]]
    print(result)
    return (result)


def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    print(f'pred_array: {pred_array}')
    result = gesture_names[np.argmax(pred_array)]
    print(f'Result: {result}')
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return result, score
def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res
    
    #starts the webcam, uses it as video source
camera = cv2.VideoCapture(0) #uses webcam for video


camera.set(10, 200)
# parameters
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 30  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variableslt
isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyboard simulator works



while camera.isOpened():
    #ret returns True if camera is running, frame grabs each frame of the video feed
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    cv2.imshow('original', frame)

    # Run once background is captured
    if isBgCaptured == 1:
        img = remove_background(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
        # cv2.imshow('mask', img)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        # cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Add prediction and action text to thresholded image
        # cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        # cv2.putText(thresh, f"Action: {action}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))  # Draw the text
        # Draw the text
        cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255))
        cv2.putText(thresh, f"Action: {action}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255))  # Draw the text
        cv2.imshow('ori', thresh)

        # get the contours
        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

        cv2.imshow('output', drawing)

    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit all windows at any time
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
       ## b.set_light(6, on_command)
        ##time.sleep(2)
        isBgCaptured = 1
        print('Background captured')
    
    elif k == ord('r'):  # press 'r' to reset the background
        time.sleep(1)
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print('Reset background')
    
    elif k == 32:
        # If space bar pressed
        cv2.imshow('original', frame)
        # copies 1 channel BW image to all 3 RGB channels
        target = np.stack((thresh,) * 3, axis=-1)
        target = cv2.resize(target, (224, 224))
        target = target.reshape(1, 224, 224, 3)
        prediction, score = predict_rgb_image_vgg(target)
      
        if prediction == "Okay":
           
            print("hi this is an Okay")
            #Going Back
            sock.sendto("down 20".encode(encoding="utf-8"), tello1_address)
           # sock.sendto("down 20".encode(encoding="utf-8"), tello2_address)
            time.sleep(5)
            
        elif prediction =="Fist":
                     # Takeoff
            print("hi this is a Fist")
            if(takeoff_cmd==1):
                sock.sendto("takeoff".encode(encoding="utf-8"), tello1_address)
                #sock.sendto("takeoff".encode(encoding="utf-8"), tello2_address)
                time.sleep(7)
                takeoff_cmd=0
            else: 
                sock.sendto("land".encode(encoding="utf-8"), tello1_address)
                #sock.sendto("land".encode(encoding="utf-8"), tello2_address)
                takeoff_cmd=1

            
            
            
            #Going forward
            #sock.sendto("up 20".encode(encoding="utf-8"), tello_address)
            #time.sleep(5)
            
        elif prediction =="L":
        
            print("hi this is a L")
            #Going left
            #sock.sendto("left 50".encode(encoding="utf-8"), tello_address)
            #time.sleep(5)
            
        elif prediction =="Peace":
        
            print("hi this is a peace")
            #Performing a forward flip
            sock.sendto("flip f".encode(encoding="utf-8"), tello1_address)
            #sock.sendto("flip f".encode(encoding="utf-8"), tello2_address)
            time.sleep(5)
            
        elif prediction == "Palm":
        
            print("hi this is a Palm")
            #Landing
            sock.sendto("cw 360".encode(encoding="utf-8"), tello1_address)
            #sock.sendto("cw 360".encode(encoding="utf-8"), tello2_address)
            #time.sleep(5)
