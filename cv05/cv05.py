import cv2
import numpy as np
import matplotlib.pyplot as plt

def prostorove_prumerovani(image_path,mask=np.array([[1,1,1],[1,1,1],[1,1,1]])):
    #maska musí mít prostředek a velikost musí být lichá
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    starting_point = int((len(mask)-1)/2)
    result = np.zeros(image.shape, dtype=int)
    number_of_poits = sum(sum(mask))
    for y in range(starting_point,image.shape[0] - starting_point):
        for x in range(starting_point,image.shape[1] - starting_point):
            Sum = 0
            for bonusY in range(-starting_point,starting_point):
                for bonusX in range(-starting_point,starting_point):
                    Sum += image[y+bonusY, x + bonusX] * mask[bonusY + starting_point][bonusX + starting_point]
            result[y,x] = int(Sum/number_of_poits)

    plt.subplot(2,2,1)
    plt.imshow(image,cmap = "gray")
    plt.subplot(2,2,2)
    plt.imshow(np.log(np.fft.fftshift(np.abs(np.fft.fft2(image)))), cmap='jet')
    plt.subplot(2,2,3)
    plt.imshow(result,cmap="gray")
    plt.subplot(2,2,4)
    plt.imshow(np.log(np.fft.fftshift(np.abs(np.fft.fft2(result)))), cmap='jet')
    return result

def rotujíci_maska(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    result = np.zeros(image.shape, dtype=int)
    starting_point = 2
    for y in range(starting_point, image.shape[0] - starting_point):
        for x in range(starting_point, image.shape[1] - starting_point):
            rozptyl_jasu = np.zeros([3,3])
            for start_y in range(-1,1):
                for start_x in range(-1,1):
                    Sum = 0
                    if start_y == 0 and start_x == 0:
                        rozptyl_jasu[1,1] = 999999 #můžu zakomentovat pokud chci povolit rotují masku s prostřednim políčkem
                        continue
                    for sum_y in range(-1,1):
                        for sum_x in range(-1,1):
                            Sum += image[y+start_y+sum_y,x+start_x+sum_x]
                    rozptyl_jasu[start_y+1,start_x+1] = Sum
            minimum = min(rozptyl_jasu)#todo vybere okolí s min rozptylem







if __name__ == "__main__":
    image_path1 = "cv05_PSS.bmp"
    image_path2 = "cv05_robotS.bmp"
    prostorove_prumerovani(image_path2)
    plt.show()


