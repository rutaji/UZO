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

def Get_DFT(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    fft2 = np.abs(np.fft.fft2(image))
    shifted_fft2 = np.fft.fftshift(fft2)

    # Vykreslení amplitudového spektra
    plt.subplot(1, 2, 1)
    plt.imshow(np.log(fft2), cmap='jet')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.log(shifted_fft2), cmap='jet')
    plt.colorbar()

    plt.show()

def Filter(picture_path, filter_path):
    image = cv2.imread(picture_path,cv2.IMREAD_GRAYSCALE) / 255
    filter = cv2.imread(filter_path,cv2.IMREAD_GRAYSCALE) / 255
    result = cv2.filter2D(image,-1,filter)
    #todo nčíst obrázek udělat furierova transformaci a pronásobit a poté furierova transformace zpět
    spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(result))))
    plt.subplot(2,2,1)
    plt.imshow(image, cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(result, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.imshow(filter, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(spectrum, cmap='jet')
    plt.show()


if __name__ == "__main__":

    # region correction
    etanol1 = "cv04_e01.bmp"
    file1 = "cv04_f01.bmp"
    etanol2 = "cv04_e02.bmp"
    file2 = "cv04_f02.bmp"

    light_correction(file1, etanol1)
    light_correction(file2, etanol2)
    # endregion

    plt.imshow(Ekvializace_histogramu("cv04_rentgen.bmp"), cmap="gray")
    plt.show()

    Get_DFT("cv04_rentgen.bmp")

    #region filter
    picture = "cv04c_robotC.bmp"
    Filter(picture, "cv04c_filtDP.bmp")
    Filter(picture, "cv04c_filtDP1.bmp")
    Filter(picture, "cv04c_filtHP.bmp")
    Filter(picture, "cv04c_filtHP1.bmp")
    #endregion





