import os
import numpy as np
import random
import cv2
from tqdm import tqdm

DATA_PATH = r'E:\CVPR\CIFAR-10-images-master\test'  # Replace with the actual path to your dataset

CATEGORIES = os.listdir(DATA_PATH)
print(CATEGORIES)

## Now to take the data in an array to train it
TRAIN_DATA = []

for c in CATEGORIES:
    path = os.path.join(DATA_PATH, c)
    for img in os.listdir(path):
        img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        img_arr = cv2.resize(img_arr, (256, 256))
        TRAIN_DATA.append([img_arr, CATEGORIES.index(c)])

print(len(TRAIN_DATA))
random.shuffle(TRAIN_DATA)

for item in TRAIN_DATA[:10]:
    print(item[0].shape, item[1])

# To display the first image in TRAIN_DATA
import matplotlib.pyplot as plt
plt.imshow(TRAIN_DATA[0][0], cmap='gray')
plt.show()

# Test image
test_img = cv2.imread('/path/to/your/test/image', cv2.IMREAD_GRAYSCALE)
test_img = cv2.resize(test_img, (256, 256))
plt.imshow(test_img, cmap='gray')
plt.show()

for img, label in TRAIN_DATA[:5]:
    print(label)

L1_dist = []
for img, label in TRAIN_DATA:
    sub_abs = np.abs(test_img - img)
    L1_dist.append([np.sum(sub_abs), label])

sorted_arr = sorted(L1_dist, key=lambda x: x[0])
print(sorted_arr[:10])
print(L1_dist[:10])



