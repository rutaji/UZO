import cv2
import matplotlib.pyplot as plt
import numpy as np
import barviccv09


def Thresholding(image,threshold):
    for y in range(0,image.shape[0]):
        for x in range(0,image.shape[1]):
            if image[y,x] < threshold:
                image[y,x] = 0
            else :
                image[y,x] = 1
    return image



if __name__ == "__main__":
    image_path1 = "cv09_rice.bmp"
    original = cv2.imread(image_path1)
    gray_image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    filterSize = (3, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       filterSize)

    tophat_img = cv2.morphologyEx(gray_image,
                                  cv2.MORPH_TOPHAT,
                                  kernel,iterations=7)


    plt.subplot(2,2,1)
    plt.imshow(gray_image,cmap="gray")
    plt.title("original")
    plt.subplot(2,2,2)
    plt.imshow(tophat_img, cmap="gray")
    plt.title("tophat")
    plt.subplot(2,2,3)
    plt.hist(gray_image.flatten(),range(250))
    plt.subplot(2,2,4)
    plt.hist(tophat_img.flatten(),range(150))
    plt.show()


    original_threshold = 120
    tophat_threshold = 50

    threshold_image = Thresholding(gray_image,original_threshold)
    threshold_image_tophat = Thresholding(tophat_img,tophat_threshold)

    plt.subplot(2, 2, 1)
    plt.imshow(threshold_image, cmap="gray")
    plt.title("original thres.")
    plt.subplot(2, 2, 2)
    plt.imshow(threshold_image_tophat, cmap="gray")
    plt.title("tophat thres.")
    plt.subplot(2, 2, 3)
    plt.hist(threshold_image.flatten())
    plt.subplot(2, 2, 4)
    plt.hist(threshold_image_tophat.flatten())
    plt.show()

    barviccv09.main(threshold_image)
    number_of_grains_original = barviccv09.Get_number_of_areas_bigger_than(91)
    centroids_original = barviccv09.DrawCentroid(original,91)
    print(f"Number of rice grains without tophat: {number_of_grains_original}")

    barviccv09.Reset()
    barviccv09.main(threshold_image_tophat)
    number_of_grains_original_tophat = barviccv09.Get_number_of_areas_bigger_than(91)
    centroids_tophat = barviccv09.DrawCentroid(original,91)
    print(f"Number of rice grains with tophat: {number_of_grains_original_tophat}")

    plt.subplot(1,2,1)
    plt.imshow(centroids_original)
    plt.title("original")
    plt.subplot(1,2,2)
    plt.title("tophat")
    plt.imshow(centroids_tophat)
    plt.show()










