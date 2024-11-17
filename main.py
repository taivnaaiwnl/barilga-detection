# Сангуудыг оруулаж ирэв.
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
# Зургийн хавтасны замууд
hagaralttai = 'C:/Users/97699/barila/hagaralttai'  # Хагаралтай зургийн хавтас
hagaraltgui = 'C:/Users/97699/barila/hagaraltgui'  # Хагаралгүй зургийн хавтас
all_images = [hagaralttai, hagaraltgui]  # Хавтаснуудыг жагсаалтанд оруулав
# Хавтас бүр дэх зургийн тоог тоолно
hagaralttai_images = len(os.listdir(hagaralttai))
hagaraltgui_images = len(os.listdir(hagaraltgui))
total_images = hagaraltgui_images + hagaralttai_images

print(f'Hagaralttai: {hagaralttai_images}')  # Хагаралтай зурагны тоог хэвлэх
print(f'Hagaraltgui: {hagaraltgui_images}')  # Хагаралгүй зурагны тоог хэвлэх

# Хагаралтай зургийг 4x4 сүлжээнд харуулах
rows, cols = 4, 4
img_count = 0
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 15))

# Хагаралтай зургийг 4x4 сүлжээнд харуулах
for i in range(rows):
    for j in range(cols):        
        if img_count < hagaralttai_images:
            axes[i, j].imshow(
                cv2.imread(os.path.join(hagaralttai, os.listdir(hagaralttai)[img_count]))
            )
            img_count += 1

img_count = 0
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 15))

# Хагаралгүй зургийг 4x4 сүлжээнд харуулах
for i in range(rows):
    for j in range(cols):        
        if img_count < hagaraltgui_images:
            axes[i, j].imshow(
                cv2.imread(os.path.join(hagaraltgui, os.listdir(hagaraltgui)[img_count]))
            )
            img_count += 1

# Зураг боловсруулах: Хэлбэрүүдийг (контур) шинжлэх
all_data = []
img, sharpen, blurred, gray, threshInv, contours = [], [], [], [], [], []
# Зарим зураг сонгож боловсруулалт хийх
img.append(cv2.imread(os.path.join(hagaralttai, os.listdir(hagaralttai)[400])))
img.append(cv2.imread(os.path.join(hagaralttai, os.listdir(hagaralttai)[500])))
img.append(cv2.imread(os.path.join(hagaraltgui, os.listdir(hagaraltgui)[400])))
img.append(cv2.imread(os.path.join(hagaraltgui, os.listdir(hagaraltgui)[500]))) 
# Ширээсний фильтрийг тодорхойлох
kernel = np.array([[-1, -1, -1], [-1, 11, -1], [-1, -1, -1]])
for i in range(len(img)):
    sharpen.append(cv2.filter2D(img[i], -1, kernel))  # Ширээс хийх
    blurred.append(cv2.GaussianBlur(sharpen[i], (3, 3), 0))  # Гауссын бүдгэрэлт
    gray.append(cv2.cvtColor(blurred[i], cv2.COLOR_BGR2GRAY))  # Саарал өнгө
    _, threshInv_ = cv2.threshold(gray[i], 230, 255, cv2.THRESH_BINARY_INV)  # Түвшинд оруулах
    threshInv.append(threshInv_)
    contours_, _ = cv2.findContours(threshInv[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_len_cnt = max([len(x) for x in contours_]) if contours_ else 0  # Хамгийн урт контурыг олох
    contours.append(max_len_cnt)
    
all_data.append(img)
all_data.append(sharpen)
all_data.append(blurred)
all_data.append(gray)
all_data.append(threshInv)
all_data.append(contours)

# Боловсруулалтын үе шатыг харуулах
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(25, 25))
for i in range(4):
    for j in range(5):
        axes[i][j].imshow(all_data[j][i])


# Хагарал илрүүлэх функц тодорхойлох
def detect_crack(image_name, contour_threshold):
    image = cv2.imread(image_name)
    kernel = np.array([[-1, -1, -1], [-1, 11, -1], [-1, -1, -1]])
    sharpen_image = cv2.filter2D(image, -1, kernel)
    blurred = cv2.GaussianBlur(sharpen_image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, threshInv = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshInv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_len_cnt = max([len(x) for x in contours]) if contours else 0
    return max_len_cnt < contour_threshold

class_result = {}
mintot = float('inf')
lp, ln = 0, 0

contour_thresholds = []
accuracies = []
precisions = []
recalls = []
f1_scores = []

for contour_threshold in range(40, 65, 5):
    for classes in all_images:
        class_count = 0
        count = 5000
        for files in os.listdir(classes):
            image_name = os.path.join(classes, files)
            if random.randint(0, 2) == 2:
                continue
            crack_status = detect_crack(image_name, contour_threshold)
            if crack_status:
                class_count += 1
            count -= 1
            if count == 0:
                break

        class_result[os.path.basename(classes)] = class_count

    tot = class_result['hagaralttai'] + 5000 - class_result['hagaraltgui']
    if tot < mintot:
        mintot = tot
        lp = 5000 - class_result['hagaralttai']
        ln = class_result['hagaraltgui']
        print(contour_threshold, tot, class_result)

    acur = (lp + ln) / 10000 * 100
    pre = lp / (lp + 5000 - ln) * 100
    recall = lp / 5000 * 100
    f1 = (2 * pre * recall) / (pre + recall)

    contour_thresholds.append(contour_threshold)
    accuracies.append(acur)
    precisions.append(pre)
    recalls.append(recall)
    f1_scores.append(f1)

    print(f'Contour Threshold: {contour_threshold}, Accuracy: {acur}, Precision: {pre}, Recall: {recall}, F1 Score: {f1}')
def detect_crack(image_name, contour_threshold):
    """
    Detects whether an image contains a crack based on contour analysis.

    Parameters:
        image_name (str): The path to the image file.
        contour_threshold (int): The threshold for maximum contour length to classify the image.

    Returns:
        has_crack (bool): True if the image is classified as hagaralttai (with crack), False otherwise.
        image (numpy.ndarray): The original image.
        sharpen_image (numpy.ndarray): The sharpened image.
        blurred (numpy.ndarray): The blurred image.
        gray (numpy.ndarray): The grayscale image.
        threshInv (numpy.ndarray): The thresholded image.
        contours (list): List of contours found in the image.
    """
    image = cv2.imread(image_name)
    if image is None:
        print(f"Error: Unable to read image '{image_name}'")
        return None, None, None, None, None, None, None

    kernel = np.array([[-1, -1, -1],
                       [-1, 11, -1],
                       [-1, -1, -1]])
    sharpen_image = cv2.filter2D(image, -1, kernel)

    blurred = cv2.GaussianBlur(sharpen_image, (3, 3), 0)

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    _, threshInv = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(threshInv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_len_cnt = max([len(x) for x in contours]) if contours else 0

    has_crack = max_len_cnt < contour_threshold

    return has_crack, image, sharpen_image, blurred, gray, threshInv, contours

def classify_image(image_name, contour_threshold):
    """
    Classifies an image as hagaralttai (with crack) or hagaraltgui (without crack).

    Parameters:
        image_name (str): The path to the image file.
        contour_threshold (int): The threshold for maximum contour length to classify the image.

    Returns:
        None
    """
    result = detect_crack(image_name, contour_threshold)
    if result[0] is None:
        return
    has_crack, image, sharpen_image, blurred, gray, threshInv, contours = result

    if has_crack:
        print('The image is classified as hagaralttai (with crack).')
    else:
        print('The image is classified as hagaraltgui (without crack).')

    images = [image, sharpen_image, blurred, gray, threshInv]
    titles = ['Original Image', 'Sharpened Image', 'Blurred Image', 'Grayscale Image', 'Thresholded Image']

    plt.figure(figsize=(20, 5))
    for i in range(len(images)):
        plt.subplot(1, 5, i+1)
        if len(images[i].shape) == 3:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()


image_name = '00001.jpg'

contour_threshold = 50

classify_image(image_name, contour_threshold)

plt.figure(figsize=(10, 6))
plt.plot(contour_thresholds, accuracies, label='Accuracy')
plt.plot(contour_thresholds, precisions, label='Precision')
plt.plot(contour_thresholds, recalls, label='Recall')
plt.plot(contour_thresholds, f1_scores, label='F1 Score')
plt.xlabel('Contour Threshold')
plt.ylabel('Metrics (%)')
plt.title('Model Performance Metrics')
plt.legend()
plt.grid(True)
plt.show()
