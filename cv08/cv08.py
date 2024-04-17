import cv2
import matplotlib.pyplot as plt
import numpy as np
import barvic


def Get_1_value(image,index):
    result = np.zeros((image.shape[0], image.shape[1]))
    for y in range(0,image.shape[0]):
        for x in range(0,image.shape[1]):
            result[y,x] = image[y,x,index] / 250
    return result

def Thresholding(image,threshold,bigger = 1):
    lower = 0
    if bigger == 0:
        lower = 1
    for y in range(0,image.shape[0]):
        for x in range(0,image.shape[1]):
            if image[y,x] < threshold:
                image[y,x] = bigger
            else :
                image[y,x] = lower
    return image

def Open(image,mask):
    result = cv2.erode(image,mask,iterations=1)
    result = cv2.dilate(result,mask,iterations=1)
    return result




if __name__ == "__main__":



    image_path1 = "cv08_im1.bmp"
    original = cv2.imread(image_path1)
    image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    plt.subplot(1, 2, 1)
    plt.imshow(image,cmap="gray")
    plt.subplot(1, 2, 2)
    plt.hist(image.flatten())
    plt.show()
    threshold = 130
    bin_image = Thresholding(image,threshold)
    mask = np.array([[0, 0, 0, 1, 0, 0, 0],
                     [0, 1, 1, 1, 1, 1, 1],
                     [0, 1, 1, 1, 1, 1, 1],
                     [0, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [0, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
    open_image = Open(bin_image, mask)
    colored_image, color_stats = barvic.main(open_image)

    plt.subplot(2,2,1)
    plt.imshow(bin_image,cmap="gray")
    plt.title("bin image")
    plt.subplot(2, 2, 2)
    plt.imshow(open_image, cmap="gray")
    plt.title("open image")
    plt.subplot(2, 2, 3)
    plt.imshow(colored_image, cmap="gray")
    plt.title("colored image")
    plt.subplot(2,2,4)
    plt.title("original image with centroid")
    plt.imshow(barvic.DrawCentroid(original))
    plt.show()

    barvic.Reset()
    original2 = cv2.imread("cv08_im2.bmp")
    original2 = cv2.cvtColor(original2,cv2.COLOR_BGR2RGB)
    YCb = cv2.cvtColor(original2, cv2.COLOR_RGB2YCR_CB)
    image_CR = Get_1_value(YCb, 1)
    image_CR_threshold = Thresholding(image_CR, 0.55, 0)
    image_CR_paint,_ = barvic.main(image_CR_threshold)
    image_cr_centroid = barvic.DrawCentroid(original2.copy())
    plt.subplot(2, 2, 1)
    plt.imshow(original2)
    plt.subplot(2, 2, 2)
    plt.imshow(image_CR_threshold,cmap="gray")
    plt.subplot(2, 2, 3)
    plt.imshow(image_CR_paint)
    plt.subplot(2, 2, 4)
    plt.imshow(image_cr_centroid)
    plt.show()

