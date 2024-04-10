import cv2
import numpy as np
import matplotlib.pyplot as plt


equivalent = list()
color_stats = dict()
class Color_stats:
    y = 0
    x = 0
    count = 0
    def add(self,y,x):
        self.y += y
        self.x += x
        self.count += 1

def Thresholding(image,threshold):
    for y in range(0,image.shape[0]):
        for x in range(0,image.shape[1]):
            if image[y,x] < threshold:
                image[y,x] = 1
            else :
                image[y,x] = 0
    return image

# vrátí seznam všech sousedů
def get_neighbours(image,y,x):
    results = set()
    for iy in range(max(0,y-1),y+1):
        for ix in range(max(0,x-1),min(image.shape[1],x+2)):
            if image[iy,ix] != 0 and image[iy,ix] != 1:
                results.add(image[iy,ix])
    return results

# přidá sousedící barvy do seznamu
def add_to_equivalent(neighbours):
    selected_lists = list()
    final_group = set()
    for neighbour in neighbours:
        found = False
        for group in equivalent:
            if neighbour in group: #Barva už je v jednom listu => vybere list a spojí s ostatními vybranými listy listy
                if group not in selected_lists:
                    selected_lists.append(group)
                found = True
                break
        if not found: # barva ještě není v seznamu
            final_group.add(neighbour)

    if len(selected_lists) > 1 or not found: #pokud byla nalezena nová barva anebo 2 či více listů potřebují spojit
        for selected_list in selected_lists:
            equivalent.remove(selected_list)
            final_group = final_group.union(selected_list)
        equivalent.append(final_group)

def get_color(value):
    for i in range(0,len(equivalent)):
        if value in equivalent[i]:
            return i+2
    raise Exception()


if __name__ == "__main__":
    image_path = "cv07_segmentace.bmp"
    image_path_test = "cv07_barveni.bmp"
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
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
    threshold = 104
    #binární obraz po segmentaci
    thresholded_image = Thresholding(gray_image,threshold)
    plt.imshow(thresholded_image, cmap="gray")
    plt.show()
    # první průchod barvení
    highest_number = 2
    for y in range(0, thresholded_image.shape[0]):
        for x in range(0, thresholded_image.shape[1]):
            if(thresholded_image[y,x] == 0):
                continue
            neighbours = get_neighbours(thresholded_image,y,x)
            if len(neighbours) == 0:
                thresholded_image[y,x] = highest_number
                equivalent.append([highest_number])
                highest_number += 1
                continue
            if len(neighbours) > 1:
                add_to_equivalent(neighbours.copy())
            thresholded_image[y, x] = neighbours.pop()
    plt.imshow(thresholded_image, cmap="gray")
    plt.show()
    for i in range(2, len(equivalent)+2):
        color_stats[i] = Color_stats()
    # druhý průchod barvení
    for y in range(0, thresholded_image.shape[0]):
        for x in range(0, thresholded_image.shape[1]):
            if (thresholded_image[y, x] == 0):
                continue
            color = get_color(thresholded_image[y, x])
            thresholded_image[y, x] = color
            color_stats[color].add(y,x)
    all_val = 0
    #výpis těžiště
    for color_stat in color_stats.values():
        xt = color_stat.x / color_stat.count
        yt = color_stat.y / color_stat.count
        val = 1
        if color_stat.count > 4000:
            val = 5
        print(f"těžiště y: {yt} x: {xt} hodnota: {val}")
        thresholded_image = cv2.circle(thresholded_image, (int(xt), int(yt)), radius=0, color=(0, 0, 255), thickness=4)
        all_val += val
    print(f"celková hodnota {all_val}")
    plt.imshow(thresholded_image)
    plt.show()

