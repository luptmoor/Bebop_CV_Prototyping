import os
import cv2
import matplotlib.pyplot as plt


folder = "Labelled"

for imagename in os.listdir(folder):
    path = folder + "/" + imagename

    image = cv2.imread(path)
    plt.figure()
    RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    RGB = cv2.rotate(RGB, cv2.ROTATE_90_COUNTERCLOCKWISE)

    RGB[:, 520//3, 0] = RGB[:, 2*520//3, 0] = 255

    plt.imshow(RGB)
    plt.show(block=False)

    label = input('Enter label: ')
    os.rename(path, label + '__' + imagename)

    plt.close()