import math

import cv2
import numpy as np

def Rotate(picture,angle):
    result = np.zeros((picture.shape[0]*2,picture.shape[1]*2,3),dtype=int)
    picture_Y = picture.shape[0]
    picture_X = picture.shape[1]
    cosA = math.cos(angle)
    sinA = math.sin(angle)
    for x in range(0,picture_X):
        for y in  range(0,picture_Y):
            newX =  int(x * cosA + y * sinA)
            newY = int( -x * sinA + y* cosA)
            newY += int(picture_Y/2)
            newX += int(picture_X/2)
            result[newY,newX,0] = picture[y,x,0]
            result[newY,newX,1] = picture[y,x,1]
            result[newY,newX,2] = picture[y,x,2]
    cv2.imshow("result",result)
    return result

if __name__ == "__main__":
    picture_path = "cv03_robot.bmp"
    picture = cv2.imread(picture_path)

    Rotate(picture,20)
    cv2.waitKey()