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
            for bonusY in range(-starting_point,starting_point+1):
                for bonusX in range(-starting_point,starting_point+1):
                    Sum += image[y+bonusY, x + bonusX] * mask[bonusY + starting_point][bonusX + starting_point]
            result[y,x] = int(Sum/number_of_poits)

    fft = np.fft.fft2(image)
    fft_plot = np.log(np.abs(np.fft.fftshift(fft)))
    fft_result = np.fft.fft2(result)
    fft_plot_result = np.log(np.abs(np.fft.fftshift(fft_result)))
    plt.subplot(2,2,1)
    plt.imshow(image,cmap = "gray")
    plt.subplot(2,2,2)
    plt.imshow(fft_plot, cmap='jet')
    plt.subplot(2,2,3)
    plt.imshow(result,cmap="gray")
    plt.subplot(2,2,4)
    plt.imshow(fft_plot_result, cmap='jet')
    plt.show()
    return result

def rotujíci_maska(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    result = np.zeros(image.shape, dtype=int)
    starting_point = 2
    for y in range(starting_point, image.shape[0] - starting_point):
        for x in range(starting_point, image.shape[1] - starting_point): # pro každý pixel
            rozptyl_jasu = np.zeros([3,3])
            for start_y in range(-1,1+1):
                for start_x in range(-1,1+1): #každá rotace masky
                    Sum = 0
                    if start_y == 0 and start_x == 0:
                        rozptyl_jasu[1,1] = 999999 #můžu zakomentovat pokud chci povolit rotují masku s prostřednim políčkem
                        continue
                    for sum_y in range(-1,1+1):
                        for sum_x in range(-1,1+1):
                            Sum += image[y+start_y+sum_y,x+start_x+sum_x]
                    stredni_hodnata = Sum/9
                    rozptyl = 0
                    for sum_y in range(-1,1+1):
                        for sum_x in range(-1,1+1):
                            rozptyl += abs(stredni_hodnata - image[y+start_y+sum_y,x+start_x+sum_x])
                    rozptyl_jasu[start_y+1,start_x+1] = rozptyl
            min_rozptyl = np.argmin(rozptyl_jasu)
            indexY = int(min_rozptyl / 3) - 1
            indexX = int(min_rozptyl % 3) - 1
            result_pixel = 0
            for sum_y in range(-1, 1+1):
                for sum_x in range(-1, 1+1):
                    result_pixel += image[y+sum_y + indexY,x+sum_x + indexX]
            result_pixel /= 9
            result[y,x] = result_pixel

    fft = np.fft.fft2(image)
    fft_plot = np.log(np.abs(np.fft.fftshift(fft)))
    fft_result = np.fft.fft2(result)
    fft_plot_result = np.log(np.abs(np.fft.fftshift(fft_result)))
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.subplot(2, 2, 2)
    plt.imshow(fft_plot, cmap='jet')
    plt.subplot(2, 2, 3)
    plt.imshow(result, cmap="gray")
    plt.subplot(2, 2, 4)
    plt.imshow(fft_plot_result, cmap='jet')
    plt.show()
    return result

def median(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    starting_point = 1
    result = np.zeros(image.shape, dtype=int)
    for y in range(starting_point, image.shape[0] - starting_point):
        for x in range(starting_point, image.shape[1] - starting_point):
            Sum = list()
            for bonusY in range(-starting_point, starting_point + 1):
                for bonusX in range(-starting_point, starting_point + 1):
                    Sum.append(image[y+bonusY, x + bonusX])
            Sum.sort()
            result[y,x] = Sum[int(len(Sum)/2)]

    fft = np.fft.fft2(image)
    fft_plot = np.log(np.abs(np.fft.fftshift(fft)))
    fft_result = np.fft.fft2(result)
    fft_plot_result = np.log(np.abs(np.fft.fftshift(fft_result)))
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.subplot(2, 2, 2)
    plt.imshow(fft_plot, cmap='jet')
    plt.subplot(2, 2, 3)
    plt.imshow(result, cmap="gray")
    plt.subplot(2, 2, 4)
    plt.imshow(fft_plot_result, cmap='jet')
    plt.show()
    return result


if __name__ == "__main__":
    image_path1 = "cv05_PSS.bmp"
    image_path2 = "cv05_robotS.bmp"
    prostorove_prumerovani(image_path2)
    rotujíci_maska(image_path2)
    median(image_path2)



