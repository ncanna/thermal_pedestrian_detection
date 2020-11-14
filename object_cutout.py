import os, glob, shutil
import cv2 as cv
from lxml import etree, objectify

# Annotations path
annotations = glob.glob('annotations-xml/set*')
annotations = sorted(annotations,key = lambda x:x[::-2])
annotations_target = os.path.basename(annotations[2])

set_num = 0

# Sets path
sets = glob.glob('Sets/set*')
sets = sorted(sets,key = lambda x:x[::-2])
target_set = os.path.basename(sets[0])

main_folder = os.path.dirname(os.path.abspath(__file__))

ppl_dir = main_folder + "/People"
per_dir = main_folder + "/Person"
cyc_dir = main_folder + "/Cyclist"

# Create folders
if os.path.exists(ppl_dir):
    shutil.rmtree(ppl_dir)
    os.makedirs(ppl_dir)
else:
    os.makedirs(ppl_dir)

if os.path.exists(per_dir):
    shutil.rmtree(per_dir)
    os.makedirs(per_dir)
else:
    os.makedirs(per_dir)

if os.path.exists(cyc_dir):
    shutil.rmtree(cyc_dir)
    os.makedirs(cyc_dir)
else:
    os.makedirs(cyc_dir)

# Loop through every set in annotations-xml
for directory in annotations:

    # Get relative path
    directory_path = os.path.basename(directory)
    annotation_videos = sorted(glob.glob(str(directory)+'/V*'))
    #print("Set annotation path: " + str(directory_path))

    # Loop through every video in every set in annotations-xml
    for video_dir in annotation_videos:
        video_annotation_path = os.path.basename(video_dir)
        # Loop through every set in Sets
        for subset in sets:
            subset_path = os.path.basename(subset)
            # print("Subset " + str(subset_path))
            # If set names match
            if subset_path == directory_path:
                print("Video annotation path: " + str(directory_path) + "/" + str(video_annotation_path))
                # print("Matching set path found for: " + str(subset_path))
                xml_files = sorted(glob.glob(str(video_dir) + '/*.xml'))
                # print(xml_files)
                sets_videos = sorted(glob.glob(str(subset) + '/V*/lwir'))
                # print(sets_videos)

                # Run script only on video with appropriate index
                set_level_base = "Sets/" + str(directory_path) + "/" + str(video_annotation_path) + "/lwir"
                # print(set_level_base)
                for set_video in sets_videos:
                    # print(set_level_base)
                    # print("set vid " + set_video)
                    if os.path.normpath(set_video) == os.path.normpath(set_level_base):
                        print(set_video)
                        # Get absolute path of annotated directory
                        # print("Image Files Path: " + str(set_video))
                        set_lwir_path = os.path.join(main_folder, set_video)

                        # Make Annotated Images Folder if Not Exists
                        abs_video_path = os.path.join(main_folder, set_video)[:-5]
                        abs_lwir_images_path = abs_video_path + "/lwir"

                        for image in os.listdir(set_lwir_path):
                            cv_img = cv.imread(set_lwir_path + "/" + image)
                            try:
                                img_name = os.path.splitext(image)[0]

                                print("Image Base: " + img_name)
                                # Find matching image to annotation based on base name
                                for anno in xml_files:
                                    # Get basename and look for a match
                                    anno_name = os.path.splitext(os.path.basename(anno))[0]
                                    # print("Annotation Base: " + str(anno_name))

                                    ppl_count = 0
                                    per_count = 0
                                    cyc_count = 0

                                    if img_name == anno_name:
                                        print("Match for Base: " + anno_name)
                                        # break  # Break at a match and anno = matching xml
                                        # Get objects in xml annotation
                                        anno_tree = etree.parse(anno)

                                        for element in anno_tree.iter():
                                            if element.tag == "object":
                                                obj_type = element[0].text
                                                print(obj_type)
                                                xmin = int(float(element[1][0].text))
                                                xmax = int(float(element[1][2].text))
                                                ymin = int(float(element[1][1].text))
                                                ymax = int(float(element[1][3].text))

                                                # determine obj type
                                                if obj_type == "cyclist":
                                                    cyc_count += 1
                                                    target_dir = cyc_dir
                                                    img_end_name = obj_type + "_" + str(cyc_count)
                                                elif obj_type == "people":
                                                    ppl_count += 1
                                                    target_dir = ppl_dir
                                                    img_end_name = obj_type + "_" + str(ppl_count)
                                                elif obj_type == "person":
                                                    per_count += 1
                                                    target_dir = per_dir
                                                    img_end_name = obj_type + "_" + str(per_count)
                                                elif obj_type == "person?":
                                                    per_count += 1
                                                    target_dir = per_dir
                                                    img_end_name = obj_type + "_" + str(per_count)

                                                cut_img = cv_img[ymin:ymax, xmin:xmax]
                                                cv.imshow("cropped", cut_img)
                                                cv.waitKey(0)

                                                # get name and target directory, then write to it

                                                fin_name = str(subset) + "_" + str(set_video) + "_" + img_name + "_" + img_end_name + ".jpg"
                                                target_dir = target_dir + "/" + fin_name
                                                cv.imwrite(target_dir, cut_img)
                                            else:
                                                pass

                                    else:
                                        pass
                            except Exception as e:
                                print(e)
                                print("Error when processing: " + str(image))
                    else:
                        pass
            else:
                pass
            set_num += 1
