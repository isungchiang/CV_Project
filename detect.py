from utils import detector_utils as detector_utils
import cv2
import numpy as np
import tensorflow as tf
import datetime
import argparse
import time
import os
import myCNN as cnn

score_thresh = 0.15
im_width = 640
im_height = 480
num_hands_detect = 1
center = (640 // 2, 480 // 2)

H1,H2 = 0,20
S1,S2 = 50,200

left = 0
right = 0
top = 0
bottom = 0

trackingMode = False
detectingMode = False

saveImg = False
sample = 155
counter = 0
path = ''

mod = 0
lastgesture = -1
guessGesture = False
lastgesturetime = datetime.datetime.now()

detection_graph, sess = detector_utils.load_inference_graph()
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
lastTime = datetime.datetime.now()


def saveROIImg(img):
    global counter, gestname, path, saveImg, lastTime
    if counter > (sample - 1):
        # Reset the parameters
        saveImg = False
        gestname = ''
        counter = 0
        lastTime = datetime.datetime.now()
        return
    currentTime = datetime.datetime.now()
    if (currentTime - lastTime).seconds > 0.2:
        counter = counter + 1
        name = gestname + str(counter)
        print("Saving img:", name)
        cv2.imwrite(path + name + 'M' + ".png", img)

        ano = cv2.flip(img,3)
        cv2.imwrite(path + name + 'F' + ".png", ano)
        # M = cv2.getRotationMatrix2D((100, 100), 90, 1)
        # rotated90 = cv2.warpAffine(img, M, (200, 200))
        # cv2.imwrite(path + name + 'R' + ".png", rotated90)
        #
        # M = cv2.getRotationMatrix2D((100, 100), 270, 1)
        # rotated270 = cv2.warpAffine(img, M, (200, 200))
        # cv2.imwrite(path + name + 'L' + ".png", rotated270)

        lastTime = datetime.datetime.now()


def binaryMask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    # blur = cv2.bilateralFilter(roi,9,75,75)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ret, res = cv2.threshold(blur, minValue, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)
    return res


def skinMask(frame):
    low_range = np.array([H1, S1, 100], dtype=np.uint8)
    upper_range = np.array([H2, S2, 255], dtype=np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)
    mask = cv2.erode(mask, skinkernel, iterations=1)
    mask = cv2.dilate(mask, skinkernel, iterations=1)
    # blur
    mask = cv2.GaussianBlur(mask, (15, 15), 1)
    # cv2.imshow("Blur", mask)
    # bitwise and mask original frame
    res = cv2.bitwise_and(frame, frame, mask=mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # ret, res = cv2.threshold(res, 70, 255, cv2.THRESH_OTSU)
    return res


# draw the detected bounding boxes on the images
# You can modify this to also draw a label.
def draw_box_on_image(scores, boxes, frame):
    global left, right, bottom, top
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (int(boxes[i][1] * im_width), int(boxes[i][3] * im_width),
                                          int(boxes[i][0] * im_height), int(boxes[i][2] * im_height))
            p1 = (left, top)
            p2 = (right, bottom)
            cv2.rectangle(frame, p1, p2, (77, 255, 9), 3, 1)
            return True
    return False


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, im_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, im_height)
    # max number of hands we want to detect/track
    mode = int(input('Train(1)/Detect(2)'))
    if mode == 1:
        print('Start Training')
        mod = cnn.loadCNN(-1)
        cnn.trainModel(mod)
        input("Press any key to continue")
    elif mode == 2 or mode == 3:
        print("Will load default weight file")
        mod = cnn.loadCNN(0)
        while True:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            ret, frame = cap.read()
            frame = cv2.flip(frame, 3)
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")

            if trackingMode:
                # actual detection
                boxes, scores = detector_utils.detect_objects(
                    frame, detection_graph, sess)

                # draw bounding boxes
                hashand = draw_box_on_image(scores, boxes, frame)
                cv2.imshow('Source', cv2.cvtColor(
                    frame, cv2.COLOR_RGB2BGR))
                if detectingMode:
                    resized_roi = np.zeros((200, 200), np.uint8)
                    filted_roi = np.zeros((200, 200), np.uint8)
                    if hashand:
                        roi = frame[top:bottom, left:right]
                        resized_roi = cv2.cvtColor(cv2.resize(roi, (200, 200)), cv2.COLOR_RGB2BGR)
                        filted_roi = skinMask(resized_roi)
                        if saveImg:
                            saveROIImg(filted_roi)
                        if guessGesture:
                            currentGestureTime = datetime.datetime.now()
                            if (currentGestureTime - lastgesturetime).seconds > 2:
                                retgesture = cnn.guessGesture(mod, filted_roi)
                                if retgesture != -1:
                                    print(cnn.output[retgesture])
                                else:
                                    print('None of Them')
                                guessGesture = False
                                lastgesturetime = datetime.datetime.now()
                    cv2.imshow('source_roi', resized_roi)
                    cv2.imshow('filter_soi', filted_roi)
            else:
                cv2.imshow('Source', cv2.cvtColor(
                    frame, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(5) & 0xFF
            if key == 27:
                break
            elif key == ord('t'):
                trackingMode = not trackingMode
                print('Tracking Mode:', trackingMode)
            elif key == ord('d'):
                detectingMode = not detectingMode
                print('Detecting Mode:', detectingMode)

            elif key == ord('s'):
                saveImg = not saveImg
                print('Saving Mode:', saveImg)

            ## Use n key to enter gesture name
            elif key == ord('n'):
                gestname = input("Enter the gesture folder name: ")
                try:
                    os.makedirs(gestname)
                except OSError as e:
                    # if directory already present
                    if e.errno != 17:
                        print('Some issue while creating the directory named -' + gestname)

                path = "./" + gestname + "/"
            elif key == ord('g'):
                guessGesture = not guessGesture
                print('guess:',guessGesture)
            # elif key == ord('y'):
            #     H1 = H1 + 2
            #     print('Low H:',H1)
            # elif key == ord('h'):
            #     H1 = H1 - 2
            #     print('Low H:',H1)
            # elif key == ord('u'):
            #     H2 = H2 + 2
            #     print('High H:',H2)
            # elif key == ord('j'):
            #     H2 = H2 - 2
            #     print('High H:',H2)
            # elif key == ord('i'):
            #     S1 = S1 + 2
            #     print('Low S:',S1)
            # elif key == ord('k'):
            #     S1 = S1 - 2
            #     print('Low S:', S1)
            # elif key == ord('o'):
            #     S2 = S2 + 2
            #     print('High S:',S2)
            # elif key == ord('l'):
            #     S2 = S2 - 2
            #     print('High S:',S2)


        cap.release()
        cv2.destroyAllWindows()
