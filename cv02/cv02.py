import cv2
import numpy as np

"""
video_path = "cv02_hrnecek.mp4"
vzor = cv2.imread('cv02_vzor_hrnecek.bmp')
vzor = cv2.cvtColor(vzor, cv2.COLOR_BGR2HSV)
vzor_hist = cv2.calcHist([vzor],[0],None,[256],[0,256])
vzor_hist_norm =  cv2.normalize(vzor_hist, vzor_hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
sizex = vzor.shape[0]
sizey = vzor.shape[1]

#test
vidcap = cv2.VideoCapture(video_path)
success,image = vidcap.read()
hsv_image = cv2.cvtColor( image,cv2.COLOR_RGB2HSV)
backproj = cv2.calcBackProject([hsv_image], [0], vzor_hist, [0,180], scale=1)
cv2.imshow('BackProj', backproj)
cv2.waitKey(0)

"""
def Hist_and_Backproj(val):
    bins = val
    histSize = max(bins, 2)
    ranges = [0, 180]  # hue_range

    hist = cv2.calcHist([hue], [0], None, [histSize], ranges, accumulate=False)
    cv2.normalize(hist, hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    backproj = cv2.calcBackProject([hue], [0], hist, ranges, scale=1)

    cv2.imshow('BackProj', backproj)

    w = 400
    h = 400
    bin_w = int(round(w / histSize))
    histImg = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(bins):
        cv2.rectangle(histImg, (i * bin_w, h), ((i + 1) * bin_w, h - int(np.round(hist[i] * h / 255.0))), (0, 0, 255),
                     cv2.FILLED)
    cv2.imshow('Histogram', histImg)

vidcap = cv2.VideoCapture("cv02_hrnecek.mp4")
success,image = vidcap.read()
src = image
if src is None:
    print('Could not open or find the image:')
    exit(0)
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
ch = (0, 0)
hue = np.empty(hsv.shape, hsv.dtype)
cv2.mixChannels([hsv], [hue], ch)
window_image = 'Source image'
cv2.namedWindow(window_image)
bins = 25
cv2.createTrackbar('* Hue  bins: ', window_image, bins, 180, Hist_and_Backproj)
Hist_and_Backproj(bins)
cv2.imshow(window_image, src)
cv2.waitKey()

