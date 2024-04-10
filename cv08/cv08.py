import cv2
import matplotlib.pyplot as plt
import numpy as np


def Thresholding(image,threshold):
    for y in range(0,image.shape[0]):
        for x in range(0,image.shape[1]):
            if image[y,x] < threshold:
                image[y,x] = 1
            else :
                image[y,x] = 0
    return image

def Open(image,mask):
    result = cv2.erode(image,mask,iterations=1)
    result = cv2.dilate(result,mask,iterations=1)
    return result




if __name__ == "__main__":
    image_path1 = "cv08_im1.bmp"
    image = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    plt.subplot(1, 2, 1)
    plt.imshow(image,cmap="gray")
    plt.subplot(1, 2, 2)
    plt.hist(image.flatten())
    plt.show()
    threshold = 130
    bin_image = Thresholding(image,threshold)
    mask = np.array([[0, 0, 0, 1, 0, 0],
                     [0, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1],
                     [0, 1, 1, 1, 1, 1]],dtype=np.uint8)#todo nepoužívám
    plt.subplot(1,2,1)
    open_image = Open(bin_image,np.ones((7,7),dtype=np.uint8))
    plt.imshow(bin_image,cmap="gray")
    plt.title("bin image")
    plt.subplot(1,2,2)
    plt.imshow(open_image,cmap="gray")
    plt.title("open image")
    plt.show()
