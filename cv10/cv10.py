import cv2
import matplotlib.pyplot as plt
import numpy as np

import barviccv10


def Closing(image,mask):
    result = cv2.dilate(image,mask,iterations=1)
    result = cv2.erode(result,mask,iterations=1)
    return result

def ThresholdWatershet(image):
    toreturn = np.zeros(image.shape, dtype=np.uint8)
    for y in range(0,image.shape[0]):
        for x in range(0,image.shape[1]):
            if image[y,x] > 1:
                toreturn[y,x] = 1
            else:
                toreturn[y,x] = 0
    toreturn = cv2.erode(toreturn,np.ones((6, 6), np.uint8),1)
    return toreturn

def Thresholding(image,threshold,lower = 255):
    bigger = abs( lower - 255)
    toreturn = np.zeros(image.shape,dtype=np.uint8)
    for y in range(0,image.shape[0]):
        for x in range(0,image.shape[1]):
            if image[y,x] < threshold:
                toreturn[y,x] = lower
            else :
                toreturn[y,x] = bigger
    return toreturn

if __name__ == "__main__":
    image = cv2.imread("cv10_mince.jpg")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    threshold = 130
    threshold_image = Thresholding(gray_image, threshold)
    closed_image = Closing(threshold_image,np.ones((6, 6), np.uint8))
    image_background = cv2.dilate(closed_image, np.ones((6, 6), np.uint8), iterations=3)
    distance_image = cv2.distanceTransform(closed_image, cv2.DIST_L2, 5)
    image_foreground = Thresholding(distance_image, 0.7 * distance_image.max(),0)
    image_foreground = np.uint8(image_foreground)
    unknown = cv2.subtract(image_background, image_foreground)
    _, markers_image = cv2.connectedComponents(image_foreground)
    markers_image += 1
    markers_image[unknown == 255] = 0
    watershed_image = cv2.watershed(image, markers_image.copy())
    watershed_original = image.copy()
    watershed_original[watershed_image == -1] = [0, 255, 0]
    watershed_image_binary = ThresholdWatershet(watershed_image)

    plt.subplot(2,4,1)
    plt.imshow(image)
    plt.title("original")
    plt.subplot(2, 4, 2)
    plt.imshow(gray_image, cmap="gray")
    plt.title("gray")
    plt.subplot(2, 4, 3)
    plt.imshow(threshold_image, cmap="gray")
    plt.title("threshold")
    plt.subplot(2, 4, 4)
    plt.imshow(closed_image, cmap="gray")
    plt.title("closed")
    plt.subplot(2,4,5)
    plt.imshow(markers_image)
    plt.title("markers")
    plt.subplot(2,4,6)
    plt.imshow(watershed_image)
    plt.title("watershed")
    plt.subplot(2,4,7)
    plt.imshow(watershed_original)
    plt.title("borders")
    plt.subplot(2,4,8)
    plt.imshow(watershed_image_binary,cmap="gray")
    plt.title("binary wat.")
    plt.show()

    barviccv10.main(watershed_image_binary)
    barviccv10.DrawCentroid(image, 10)
    plt.imshow(image)
    plt.show()






