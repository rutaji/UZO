import cv2
import numpy as np

def calculate_center(back_projection):
    x, y = np.meshgrid(np.arange(back_projection.shape[1]), np.arange(back_projection.shape[0]))
    new_center_X = np.sum(x * back_projection) / np.sum(back_projection)
    new_center_Y = np.sum(y * back_projection) / np.sum(back_projection)
    return new_center_X, new_center_Y

if __name__ == "__main__":
    video_path = "cv02_hrnecek.mp4"
    vzor = cv2.imread('cv02_vzor_hrnecek.bmp')
    vzor = cv2.cvtColor(vzor, cv2.COLOR_BGR2HSV)
    sizey = vzor.shape[0] / 2
    sizex = vzor.shape[1] / 2
    vzor = vzor[:, :, 0]
    vzor_hist = cv2.calcHist([vzor],[0],None,[180],[0,180]).flatten()
    vzor_hist = vzor_hist/max(vzor_hist)
    vzor_hist = vzor_hist / np.max(vzor_hist)

    center_X = None
    center_Y = None
    #nahrání videa
    vidcap = cv2.VideoCapture(video_path)
    while True:
        ok, frame = vidcap.read()
        if not ok:
            break;
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = hsv_frame[:, :, 0]
        back_projection = vzor_hist[hue]
        if center_X is None:
            #first frame
            center_X, center_Y = calculate_center(back_projection)
        else:
            back_projection = back_projection[corner_y1:corner_y2, corner_x1:corner_x2]
            new_center_X, new_center_Y = calculate_center(back_projection)
            center_X = int(corner_x1 + new_center_X)
            center_Y = int(corner_y1 + new_center_Y)

        corner_x1 = abs(int(center_X - sizex))
        corner_y1 = abs(int(center_Y - sizey))
        corner_x2 = abs(int(center_X + sizex))
        corner_y2 = abs(int(center_Y + sizey))
        cv2.rectangle(frame, (corner_x1, corner_y1), (corner_x2, corner_y2), (0, 255, 0), 3)
        cv2.imshow('frame', frame)
        cv2.waitKey()



