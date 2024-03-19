import math

import cv2
import numpy as np

def Rotate(picture,angle):
    new_size=math.ceil(math.sqrt(picture.shape[0]**2 + picture.shape[1]**2))
    result = np.zeros((new_size,new_size,3),dtype=np.uint8)
    picture_Y = picture.shape[0]-1
    picture_X = picture.shape[1]-1
    cosA = math.cos(angle)
    sinA = math.sin(angle)
    cornersX = [0,int(picture_X * cosA + 0 * sinA),int(0 * cosA + picture_Y * sinA),int(picture_X * cosA + picture_Y * sinA)]
    cornersY= [0,int( -0 * sinA + picture_Y * cosA),int( -picture_X * sinA + 0 * cosA),int( -picture_X * sinA + picture_Y * cosA)]
    minX = min(cornersX)
    minY = min(cornersY)
    for x in range(0,picture_X+1):
        for y in range(0,picture_Y+1):
            newX = round(x * cosA + y * sinA)
            newY = round( -x * sinA + y * cosA)
            newY += abs(minY)
            newX += abs(minX)
            result[newY,newX,0] = picture[y,x,0]
            result[newY,newX,1] = picture[y,x,1]
            result[newY,newX,2] = picture[y,x,2]
    return result


if __name__ == "__main__":
    picture_path = "cv03_robot.bmp"
    picture = cv2.imread(picture_path)
    result = Rotate(picture,4)
    cv2.imshow("result", result)
    cv2.waitKey()
