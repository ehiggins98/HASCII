"""
Splits each scan in the current directory into individual characters, processes those images,
and saves them in a numpy matrix.
"""

import line_processor as lp
import cv2 as cv
import math
import numpy as np
from os import listdir
from os.path import isfile, join
import re

page1 = [
 '!', ')', '"', '*', '#', '+', '$', '-', '%', '.', '&', '/', '\'', ':', '(', ';'
]

page2 = [
    '<', '^', '=', '_', '>', '`', '?', '{', '@', '}', '[', '|', '\\', '~', ']', ','
]

classes = {
    ':': 62,
    ';': 63,
    '<': 64,
    '=': 65,
    '>': 66,
    '?': 67,
    '!': 68,
    '"': 69,
    '%': 70,
    '&': 71,
    '\'': 72,
    '(': 73,
    ')': 74,
    '*': 75,
    '+': 76,
    ',': 77,
    '-': 78,
    '.': 79,
    '/': 80,
    '[': 81,
    ']': 82,
    '^': 83,
    '{': 84,
    '|': 85,
    '}': 86,
    '#': 87,
    '$': 88,
    '_': 89,
    '`': 90,
    '@': 91,
    '\\': 92,
    '~': 93
}

num_contours = {
    62: 2,
    63: 2,
    64: 1,
    65: 2,
    66: 1,
    67: 2,
    68: 2,
    69: 2,
    70: 3,
    71: 1,
    72: 1,
    73: 1,
    74: 1,
    75: 1,
    76: 1,
    77: 1,
    78: 1,
    79: 1,
    80: 1,
    81: 1,
    82: 1,
    83: 1,
    84: 1,
    85: 1,
    86: 1,
    87: 1,
    88: 1,
    89: 1,
    90: 1,
    91: 1,
    92: 1,
    93: 1
}

def angle(line):
    """
    Gets the angle of a line.
    @param line: A doubly-nested tuple (or list) of endpoints. I.e. [(a1, b1), (a2, b2)]
    """
    return math.atan((line[1][1] - line[0][1]) / (line[1][0] - line[0][0]))

def add_to_sorted_list(list, line, direction):
    """
    Adds a line to the list such that the list remains sorted.
    @param list: The list of lines to which to add the line.
    @param line: The line to add to the list.
    @param direction The direction of lines in the list (either 'horizontal' or 'vertical').
    """
    dimension = 1 if direction == 'horizontal' else 0
    i = 0
    while i < len(list) and list[i][0][dimension] < line[0][dimension]:
        i += 1

    list.insert(i, line)

def contains(rect, c2):
    """
    Returns a value indicating whether contour c2 is at least partially contained in rect
    @param rect A tuple (x, y, w, h) with x, y being the coordinates of the upper-left corner and
    w, h being the width and height of the rectangle, respectively.
    @param c2 The "inner" contour.
    """
    x1, y1, w1, h1 = rect
    x2, y2, w2, h2 = cv.boundingRect(c2)

    return ((x2 > x1 and x2 < x1 + w1) + (x2 + w2 > x1 and x2 + w2 < x1 + w1) 
    + (y2 > y1 and y2 < y1 + h1) + (y2 + h2 > y1 and y2 + h2 < y1 + h1)) > 1

def process_image_step_1(img, img_class):
    """
    Finds the bounding box of the character in the given image, with a 2-pixel border.
    The Gaussian Blur is performed in order to be consistent with the processing in the EMNIST dataset.
    @param img: The image for which to process the bounding box.
    @return The filtered and blurred image and the bounding box of the character contained in it.
    """
    orig = np.copy(img)
    img = np.clip(img - 20, 0, 255)
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img = cv.inRange(img, (0.0, 0.0, 0.0),  (255.0, 255.0, 192.0))
    img = cv.GaussianBlur(img, (0, 0), 1)

    _, contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    bound_x = bound_y = bound_w = bound_h = 0

    n = num_contours[img_class]
    areas = map(lambda c: (c, cv.contourArea(c)), contours)
    n_largest = sorted(areas, key=lambda t: t[1])[-n:]
    n_largest = list(map(lambda t: t[0], n_largest))

    if len(n_largest) > 0:
        bound_x, bound_y, bound_w, bound_h = cv.boundingRect(n_largest[0])

    for c in n_largest:
        if cv.contourArea(c) < 40:
            continue
        
        x, y, w, h = cv.boundingRect(c)

        if bound_x > x:
            bound_w = bound_w + bound_x - x
            bound_x = x
        if bound_x + bound_w < x + w:
             bound_w = x + w - bound_x
        if bound_y > y:
             bound_h = bound_h + bound_y - y
             bound_y = y
        if bound_y + bound_h < y + h:
             bound_h = y + h - bound_y

    if bound_y <= 1:
        bound_y += 2
    if bound_x <= 1:
        bound_x += 2

    # there's an issue with % specifically where the edges get cut off
    if img_class == 70:
        bound_x -= 8
        bound_w += 16
        bound_y -= 5
        bound_h += 10

    bounding_rect = (bound_x-2, bound_y-2, bound_w+4, bound_h+4)

    for c in contours:
        if not contains(bounding_rect, c) and c not in n_largest:
            x, y, w, h = cv.boundingRect(c)
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), -1)

    return img, bounding_rect

def process_image_step_2(img, size, original_bounds):
    """
    Perfoms final processing. This includes centering the character in a square image, reducing its
    size to [32, 32], and scaling the image such that values are in the range [0, 255].
    @param img: The image to process.
    @param size: The maximum size, computed over all character images. This can be obtained by taking
    the max over the bounding boxes returned by process_image_step_1.
    @param original_bounds The bounds of the character in the original image.
    @return a [32, 32] image with the character centered in it.
    """
    x, y, w, h = original_bounds
    img = img[y:y+h, x:x+w]
    dim = max(int(size), 32)
    if dim % 2 == 1:
        dim += 1
    center = (int(np.size(img, axis=0)/2), int(np.size(img, axis=1)/2))
    square = np.zeros((size, size))

    for i in range(0, np.size(img, axis=0)):
        for j in range(0, np.size(img, axis=1)):
            square[math.floor(dim / 2 - center[0]) + i - 1, math.floor(dim / 2 - center[1]) + j - 1] = img[i, j]
    square = cv.resize(square, (32, 32), interpolation=cv.INTER_CUBIC)
    square *= 255 / np.max(square) if np.max(square) > 0 else 1
    return square

def remove_min_and_max_if_necessary(list, target_length, argmin_first):
    """
    Removes the first and last elements of the list if necessary. It seems kind of sketchy, but
    seems to work for the whole dataset.
    @param list: The list to process.
    @param target_length: The desired length of the list, after removing elements.
    @param argmin_first: Whether to remove the leftmost line first.
    """
    first_remove = list[0] if argmin_first else list[len(list)-1]
    then_remove = list[len(list)-2] if argmin_first else list[0]
    if len(list) > target_length:
        list.remove(first_remove)
        if len(list) > target_length:
            list.remove(then_remove)
    return list

def get_label(row, col, configuration):
    """
    Gets the label for a given image.
    @param row: The row containing the image.
    @param col: The column containing the image.
    @param configuration: Either 1 or 2, indicating whether Page 1 or Page 2 is being processed.
    """
    if configuration == 1:
        label = classes[page1[2*row + (col > 3)]]
    elif configuration == 2:
        label = classes[page2[2*row + (col > 3)]]
    return label

def find_intersect(l1, l2):
    """
    Finds the intersection of two lines
    @param l1 The first line.
    @param l2 The second line.
    @return A tuple representing (row, col) of the intersection.
    """

    # slopes of the lines
    m1 = (l1[1][1] - l1[0][1]) / (l1[1][0] - l1[0][0])
    m2 = (l2[1][1] - l2[0][1]) / (l2[1][0] - l2[0][0])

    # for numerical stability
    if m1 == 0:
        m1 += 0.0001
    if m2 == 0:
        m2 += 0.0001

    # these are derived from the equations for the lines
    y = (-m1*l1[0][0] + m1*l2[0][0] + l1[0][1] - m1*l2[0][1]/m2)*(m2/(m2-m1))
    x = 1/m1 * (y - l1[0][1] + m1*l1[0][0])

    return [int(x), int(y)]

def crop_within(img, top_border, right_border, bottom_border, left_border):
    """
    Crops the polygonal region from img bounded by the given lines.
    @param img The image from which to crop.
    @param l1: The topmost bounding line.
    @param l2: The rightmost bounding line.
    @param l3: The bottommost bounding line.
    @param l4: The leftmost bounding line.
    @return A minimally-sized rectangle containing the region of interest.
    """
    mask = np.zeros(np.shape(img))
    p1 = find_intersect(top_border, right_border)
    p2 = find_intersect(right_border, bottom_border)
    p3 = find_intersect(bottom_border, left_border)
    p4 = find_intersect(left_border, top_border)

    cv.fillConvexPoly(mask, np.array([p1, p2, p3, p4]), (1, 1, 1))
    result = np.multiply(img, mask.astype(np.uint8))

    result = result[min(p1[1], p4[1]) + 15:max(p2, p3)[1] - 15, min(p3[0], p4[0]) + 15:max(p1[0], p2[0]) - 15]
    return result

def crop_data(file, configuration):
    """
    Perform full processing on an image. Takes a raw scan and returns a numpy matrix containing
    all character images and labels from it.
    @param file: The name of the image file to process.
    @param configuration: Either 1 or 2, indicating whether Page 1 or Page 2 is being processed.
    """
    images = np.zeros((0, 32, 32))
    labels = np.zeros((0))
    lines = lp.process_lines(file)
    angle_threshold = 0.1
    img = cv.imread(file)
    vertical = []
    horizontal = []

    max_dim = 0
    bounding_boxes = []
    for l in lines:
        if lp.lineMagnitude(l[0][0], l[0][1], l[1][0], l[1][1]) > 400:
            add_to_sorted_list(horizontal, l, 'horizontal') if abs(angle(l) - 0) < angle_threshold else add_to_sorted_list(vertical, l, 'vertical')
    count = 0

    horizontal = remove_min_and_max_if_necessary(horizontal, 9, True)
    vertical = remove_min_and_max_if_necessary(vertical, 7, False)

    if len(horizontal) != 9 or len(vertical) != 7:
        print("Incorrect number of lines detected in", file)
        print("Skipping; data would likely be incorrect")
        return
    k = 0
    for i in range(1, len(vertical)-1):
        if i != 3:
            left_line = vertical[i]
            right_line = vertical[i+1]
            for j in range(0, len(horizontal) - 1):
                count += 1
                top_line = horizontal[j]
                bottom_line = horizontal[j+1]
                cropped = crop_within(img, top_line, right_line, bottom_line, left_line)
                img_class = get_label(j, i, configuration)
                cropped, box = process_image_step_1(cropped, img_class)
                k += 1
                if np.max(cropped) > 0:
                    bounding_boxes.append((cropped, box, img_class))
                    if max(box[2], box[3]) > max_dim:
                        max_dim = max(box[2], box[3])
    for i in range(0, len(bounding_boxes)):
        img = process_image_step_2(bounding_boxes[i][0], max_dim, bounding_boxes[i][1])
        img = np.reshape(img, (1, 32, 32))
        images = np.append(images, img, axis=0)
        labels = np.append(labels, bounding_boxes[i][2])

    return images, labels

image_files = [f for f in listdir('.') if isfile(join('.', f)) and re.compile('.+\.(JPE?G|jpe?g)$').match(f)]
all_images = np.load('images.npy')
all_labels = np.load('labels.npy')
for i in range(len(image_files)):
    if i % 1 == 0:
        print(image_files[i], np.shape(all_images), np.shape(all_labels))

    if i % 20 == 0:
        np.save('images.npy', all_images)
        np.save('labels.npy', all_labels)        
    images, labels = crop_data(image_files[i], 2)
    all_images = np.append(all_images, images, axis=0)
    all_labels = np.append(all_labels, labels, axis=0)

np.save('images.npy', all_images)
np.save('labels.npy', all_labels)
