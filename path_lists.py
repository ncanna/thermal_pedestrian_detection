import csv
import os, glob, shutil
from lxml import etree, objectify

main_folder = os.path.dirname(os.path.abspath(__file__))
set_folder = os.path.join(main_folder, "Sets")
anno_folder = os.path.join(main_folder, "annotations-xml")

filename = "frame_MasterList.csv"
rows = []

# Fields
fields = ["image_path", "annotation_path"]

# Get Cutout_ID and path
for set in os.listdir(set_folder):

    target_set = os.path.join(set_folder, set)

    for video in os.listdir(target_set):

        target_lwir = os.path.join(target_set, video, "lwir")

        for img in os.listdir(target_lwir):
            temp_row = []

            # Get cutout id and paths and append to a temporary list
            # img_name = os.path.basename(img)
            # img_path = img_name[0:5] + "/" + img_name[6:10] + "/lwir/" + img_name[11:17] + ".jpg"
            # img_path = os.path.join(main_folder, "Sets", os.path.normpath(img_path))
            #print(img_path)

            img_name = os.path.basename(img)
            img_path = os.path.join(target_lwir, img)
            print(img_path)

            target_anno = os.path.join(anno_folder, set, video)

            # Find matching annotation
            for annotation in os.listdir(target_anno):
                anno_name = os.path.basename(annotation)

                # Check for a matching annotation to image
                if os.path.splitext(img_name)[0] == os.path.splitext(anno_name)[0]:
                    print(os.path.splitext(img_name)[0] + " == " + os.path.splitext(anno_name)[0])

                    anno_tree = etree.parse(os.path.join(target_anno, annotation))

                    #Check if an object is contained in the annotation and then append to row
                    for element in anno_tree.iter():
                        if element.tag == "object":
                            print("object found in " + img_name + ", " + annotation)
                            img_anno = os.path.join(target_anno, annotation)

                            # Append data
                            temp_row.append(img_path)
                            temp_row.append(img_anno)
                            rows.append(temp_row)
                            print(*temp_row)

                            break

            # Append temp list to the main csv list



# writing to csv file
with open(filename, 'w', newline='') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(rows)
