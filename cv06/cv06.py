import cv2
import numpy as np
import matplotlib.pyplot as plt





def display(image,result):
    fft = np.fft.fft2(image)
    fft_plot = np.log(np.abs(np.fft.fftshift(fft)))
    fft_result = np.fft.fft2(result)
    fft_plot_result = np.log(np.abs(np.fft.fftshift(fft_result)))
    plt.subplot(2,2,1)
    plt.imshow(image,cmap = "gray")
    plt.subplot(2,2,2)
    plt.imshow(fft_plot, cmap='jet')
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(result,cmap="jet")
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(fft_plot_result, cmap='jet')
    plt.colorbar()
    plt.show()


def ApplyMask(image,mask):
    starting_point = int((len(mask)-1)/2)
    result = np.zeros(image.shape, dtype=int)
    for y in range(starting_point,image.shape[0] - starting_point):
        for x in range(starting_point,image.shape[1] - starting_point):
            Sum = 0
            for bonusY in range(-starting_point,starting_point+1):
                for bonusX in range(-starting_point,starting_point+1):
                    Sum += image[y+bonusY, x + bonusX] * mask[bonusY + starting_point][bonusX + starting_point]
            result[y,x] = int(Sum)
    return result


def Laplaceuv(image):
    result = ApplyMask(image,np.array([[0,1,0],[1,-4,1],[0,1,0]]))
    display(image,result)
    

def sobeluv(image):
    h1 = ApplyMask(image,np.array([[1,2,1],
                                   [0,0,0],
                                   [-1,-2,-1]]))
    
    h2 = ApplyMask(image,np.array([[0,1,2],
                                   [1,0,1],
                                   [-2,-1,0]]))
    
    h3 = ApplyMask(image,np.array([[-1,0,1],
                                   [-2,0,2],
                                   [-1,0,1]]))
    
    h4 = ApplyMask(image,np.array([[-2,-1,0],
                                   [-1,0,1],
                                   [0,1,2]]))
    
    h5 = ApplyMask(image,np.array([[-1,-2,-1],
                                   [0,0,0],
                                   [1,2,1]]))
    
    h6 = ApplyMask(image,np.array([[0,-1,-2],
                                   [1,0,-1],
                                   [2,1,0]]))
    
    h7 = ApplyMask(image,np.array([[1,0,-1],
                                   [2,0,-2],
                                   [1,0,-1]]))
    
    h8 = ApplyMask(image,np.array([[2,1,0],
                                   [1,0,1],
                                   [0,-1,-2]]))
    
    result = np.abs(h1) + np.abs(h2) + np.abs(h3) + np.abs(h4) + np.abs(h5) + np.abs(h6) + np.abs(h7) + np.abs(h8)
    display(image,result)


def kirsch(image):
    h1 = ApplyMask(image, np.array([[3, 3, 3],
                                    [3, 0, 3],
                                    [-5, -5, -5]]))

    h2 = ApplyMask(image, np.array([[3, 3, 3],
                                    [-5, 0, 3],
                                    [-5, -5, 3]]))

    h3 = ApplyMask(image, np.array([[-5, 3, 3],
                                    [-5, 0, 3],
                                    [-5, 3, 3]]))

    h4 = ApplyMask(image, np.array([[-5, -5, 3],
                                    [-5, 0, 3],
                                    [3, 3, 3]]))

    h5 = ApplyMask(image, np.array([[-5, -5, -5],
                                    [3, 0, 3],
                                    [3, 3, 3]]))

    h6 = ApplyMask(image, np.array([[3, -5, -5],
                                    [3, 0, -5],
                                    [3, 3, 3]]))

    h7 = ApplyMask(image, np.array([[3, 3, -5],
                                    [3, 0, -5],
                                    [3, 3, -5]]))

    h8 = ApplyMask(image, np.array([[3, 3, 3],
                                    [3, 0, -5],
                                    [3, -5, -5]]))

    result = np.abs(h1) + np.abs(h2) + np.abs(h3) + np.abs(h4) + np.abs(h5) + np.abs(h6) + np.abs(h7) + np.abs(h8)
    display(image, result)
    



if __name__ == "__main__":
    image_path1 = "cv06_robotC.bmp"
    image = cv2.imread(image_path1,cv2.IMREAD_GRAYSCALE)
    Laplaceuv(image)
    sobeluv(image)
    kirsch(image)
    
