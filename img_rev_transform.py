import os, glob
import cv2 as cv

main_folder = os.path.dirname(os.path.abspath(__file__))

train_folder = os.path.join(main_folder, "train")

for obj_dir in train_folder:
    print(obj_dir)
    for img in obj_dir:
        print(img)
        original_img = cv.imread(img, 1)
        img_name_rev = os.path.basename(img) + "_rev.jpg"
        target_name = os.path.join(train_folder, obj_dir) + "/" + img_name_rev
        flipped = cv.flip(original_img, 1)
        cv.imwrite(target_name, flipped)
