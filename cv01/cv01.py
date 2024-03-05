import math

import cv2
import matplotlib.pyplot as plt

def Compare_Picture(picture_to_compare = "im02.jpg"):
    pictures = ["im01.jpg","im02.jpg","im03.jpg","im04.jpg","im05.jpg","im06.jpg","im07.jpg","im08.jpg","im09.jpg"]
    images = list()
    histograms = list()
    for p in pictures:
        h = cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB)
        images.append(h)
        histograms.append(cv2.calcHist([cv2.cvtColor(h,cv2.COLOR_RGB2GRAY)], [0], None, [256], [0, 256]))
    pass

    histogram_to_compare = cv2.calcHist([cv2.cvtColor(cv2.imread(picture_to_compare),cv2.COLOR_BGR2GRAY)],[0],None,[256],[0,256])
    distance = list()
    for h in histograms:
        sum = 0
        for i in range(0, len(h)):
            sum +=  abs(h[i]- histogram_to_compare[i]) ** 2
        distance.append(math.sqrt( sum))
    for i in range(1,len(distance)+1):
        min_index = distance.index(min(distance))
        plt.subplot(1, 9, i)
        fig = plt.imshow(images.pop(min_index))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        distance.pop(min_index)

    plt.show()


if __name__ == '__main__':
    Compare_Picture()



