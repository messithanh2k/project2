import cv2
import numpy as np

img_array = []
folder = "image_cut/project/"
for i in range(1, 174):
    img = cv2.imread(folder + str(i) + ".jpg")
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('project.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

