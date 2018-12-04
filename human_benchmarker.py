"""
A simple script for getting a human benchmark on EMNIST.
"""
import random as rand
import numpy as np
import cv2 as cv
import math

images = np.load('dataset\\punctuation\\all_images.npy')
labels = np.load('dataset\\punctuation\\all_labels.npy')

correct = 0
total = 0

classes = {
    62: 58,
    63: 59,
    64: 60,
    65: 61,
    66: 62,
    67: 63,
    68: 33,
    69: 34,
    70: 37,
    71: 38,
    72: 39,
    73: 40,
    74: 41,
    75: 42,
    76: 43,
    77: 44,
    78: 45,
    79: 46,
    80: 47,
    81: 91,
    82: 93,
    83: 94,
    84: 123,
    85: 124,
    86: 125,
    87: 35,
    88: 36,
    89: 95,
    90: 96,
    91: 64,
    92: 92,
    93: 126
}

for i in range(0, np.size(images, axis=0)-1):
    index = rand.randint(0, np.size(images, axis=0)-1)
    label = labels[index][0]

    if label <= 9 and label not in classes:
        label += 48
    elif label >= 10 and label <= 35 and label not in classes:
        label += 55
    elif label not in classes:
        label += 61
    else:
        for k, v in classes.items():
            if label == k:
                label = v
                break

    cv.imshow('Image', images[index])
    key = cv.waitKey(0)

    if key == 13:
        break

    if label == key:
        correct += 1
    total += 1

    if total % 100 == 0:
        print(total)

print(correct, total, correct/total)
