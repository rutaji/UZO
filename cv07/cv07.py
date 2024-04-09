import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    image = cv2.imread("cv07_segmentace.bmp", cv2.COLOR_BGR2RGB)
    gray_image = np.zeros([image.shape[0], image.shape[1]])
    for y in range(0,image.shape[0]):
        for x in range(0,image.shape[1]):
            R = int(image[y,x,0])
            G = int(image[y, x, 1])
            B = int(image[y, x, 2])
            if (R+G+B) == 0:
                gray_image[y,x] = 0
            else:
                gray_image[y,x] =(G*255)/(R+G+B)
    plt.subplot(1,3,1)
    plt.imshow(image)
    plt.subplot(1,3,2)
    plt.imshow(gray_image)
    plt.subplot(1,3,3)
    plt.hist(gray_image.flatten())
    plt.show()
    prah = 104
