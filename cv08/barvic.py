import  matplotlib.pyplot as plt
import cv2
import numpy as np

equivalent = list()

def Thresholding(image,threshold):
    for y in range(0,image.shape[0]):
        for x in range(0,image.shape[1]):
            if image[y,x] < threshold:
                image[y,x] = 1
            else :
                image[y,x] = 0
    return image

#region barvení oblastí

def get_neighbours(image,y,x):
    results = set()
    for iy in range(max(0,y-1),y+1):
        for ix in range(max(0,x-1),min(image.shape[1],x+2)):
            if image[iy,ix] != 0 and image[iy,ix] != 1:
                results.add(image[iy,ix])
    return results
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

def compute_centroid():
    for color_stat in color_stats.values():
        color_stat.xt = color_stat.x / color_stat.count
        color_stat.yt = color_stat.y / color_stat.count
def get_color(value):
    for i in range(0,len(equivalent)):
        if value in equivalent[i]:
            return i+2
    raise Exception()

#region statistiky o barvení oblastí
color_stats = dict()
class Color_stats:
    y = 0
    x = 0
    count = 0
    xt = 0
    yt = 0
    def add(self,y,x):
        self.y += y
        self.x += x
        self.count += 1
#endregion
#endregion

#region public method

def main(image):
    # první průchod barvení
    result = image.copy()
    highest_number = 2
    for y in range(0, result.shape[0]):
        for x in range(0, result.shape[1]):
            if result[y, x] == 0:
                continue
            neighbours = get_neighbours(result, y, x)
            if len(neighbours) == 0:
                result[y, x] = highest_number
                equivalent.append([highest_number])
                highest_number += 1
                continue
            if len(neighbours) > 1:
                add_to_equivalent(neighbours.copy())
            result[y, x] = neighbours.pop()
    for i in range(2, len(equivalent) + 2):
        color_stats[i] = Color_stats()
    # druhý průchod barvení
    for y in range(0, result.shape[0]):
        for x in range(0, result.shape[1]):
            if result[y, x] == 0:
                continue
            color = get_color(result[y, x])
            result[y, x] = color
            color_stats[color].add(y, x)
    compute_centroid()
    return result, color_stats





def DrawCentroid(image):
    for color_stat in color_stats.values():
        image = cv2.circle(image, (int(color_stat.xt), int(color_stat.yt)), radius=0, color=(0, 255,0), thickness=4)
    return image
