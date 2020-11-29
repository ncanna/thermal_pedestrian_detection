import csv
import os, glob, shutil

main_folder = os.path.dirname(os.path.abspath(__file__))
train_folder = os.path.join(main_folder, "train")
anno_folder = os.path.join(main_folder, "annotations-xml")

filename = "cutout_MasterList.csv"
rows = []

# Fields
fields = ["Cutout_ID", "Cutout_Path", "Annotation_Path"]

#Get Cutout_ID and path
for class_dir in os.listdir(train_folder):

    for img in os.listdir(os.path.join(train_folder, class_dir)):
        temp_row = []

        # Get cutout id and paths and append to a temporary list
        img_name = os.path.basename(img)
        img_path = os.path.abspath(img)
        temp_row.append(img_name)
        temp_row.append(img_path)

        target_anno = os.path.join(anno_folder, img_name[0:5])
        target_anno = os.path.join(target_anno, img_name[6:10])

        # Find matching annotation
        for annotation in os.listdir(target_anno):
            anno_name = os.path.basename(annotation)

            if img_name[11:17] == os.path.splitext(anno_name)[0]:

                img_anno = os.path.join(target_anno, annotation)

        # Append to temp list
        temp_row.append(img_anno)

        # Append temp list to the main csv list
        rows.append(temp_row)
        print(*temp_row)

# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    # writing the fields
    csvwriter.writerow(fields)

    # writing the data rows
    csvwriter.writerows(rows)
