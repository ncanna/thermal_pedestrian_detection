import os, glob
import cv2 as cv

cutoutDir = ""

for img in cutoutDir:
    original_img = cv.imread(img, 1)
    img_name_rev = os.path.basename(img) + "_rev.jpg"
    flipped = cv.flip(original_img, 1)
    cv.imwrite(img_name_rev, flipped)
