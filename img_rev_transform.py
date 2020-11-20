import os, glob
import cv2 as cv

#### CODE IS NOT OPERABLE ####


main_folder = os.path.dirname(os.path.abspath(__file__))

train_folder = os.path.join(main_folder, "train")

for obj_dir in os.listdir(train_folder):
    print(obj_dir)
    target_folder = os.path.join(train_folder, obj_dir)
    for img in os.listdir(target_folder):
        print(img)
        original_img = cv.imread(img, 1)

        cv.imshow("original", original_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

        img_name_rev = os.path.basename(img) + "_rev.jpg"
        target_name = target_folder + "/" + img_name_rev
        flipped = cv.flip(original_img, 1)

        cv.imshow(img_name_rev, flipped)
        cv.waitKey(0)
        cv.destroyAllWindows()

        cv.imwrite(target_name, flipped)
