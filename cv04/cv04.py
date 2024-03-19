import cv2
import matplotlib.pyplot as plt
import numpy as np




def light_correction(file_path, etanol_path):
    # načtení souboru
    file = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
    etanol = cv2.cvtColor(cv2.imread(etanol_path), cv2.COLOR_BGR2RGB)
    # načtení rozměrů
    height = file.shape[0]
    width = file.shape[1]
    # cyklus
    result = file.copy()
    for y in range(0, height):
        for x in range(0, width):
            result[y][x][0] = 255 * file[y][x][0] / etanol[y][x][0]
            result[y][x][1] = 255 * file[y][x][1] / etanol[y][x][1]
            result[y][x][2] = 255 * file[y][x][2] / etanol[y][x][2]

    # plot
    plt.subplot(1, 3, 1)
    plt.imshow(file)
    plt.subplot(1, 3, 2)
    plt.imshow(etanol)
    plt.subplot(1, 3, 3)
    plt.imshow(result)
    plt.show()
    return result

def Ekvializace_histogramu(input):
    image = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    sum = hist.cumsum()
    sum_norm = sum / float(sum[-1])
    equalized_image = np.interp(image.flatten(), bins[:-1], sum_norm * 255).reshape(image.shape)
    return equalized_image.astype(np.uint8)


if __name__ == "__main__":
    #region correction
    etanol1="cv04_e01.bmp"
    file1="cv04_f01.bmp"
    etanol2 = "cv04_e02.bmp"
    file2 = "cv04_f02.bmp"

    light_correction(file1,etanol1)
    light_correction(file2,etanol2)
    #endregion
    #region ekvalizace
    plt.imshow(Ekvializace_histogramu("cv04_rentgen.bmp"),cmap="gray")
    plt.show()
    #endregion

