import cv2
import glob
import os


def makeVideo(video):
    img_array = []

    #make a list of files:
    frames = os.listdir(video + "annotated/")
    frames = len(frames) #number of frames in the video

    #add frames into the img array
    for i in range(frames+1):
        i = str(i)
        n = 5-len(i)
        fin = video + "annotated/I"+  n*"0" + i +".jpg" #might require more adjusting depending on how the names are structured but seems to work for now
        print(fin)
        img_array.append(cv2.imread(fin))

    height,width,layers = img_array[1].shape

    result = cv2.VideoWriter(locale+"video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 15, (width, height))

    for i in range(len(img_array)):
        result.write(img_array[i])
    print("Video for: " + video + " is done.")
    result.release()


#Get Directory
sets = glob.glob('Sets/set*')
#print(sets)
videos = {}
for set in sets:
    videos[set] = glob.glob(str(set) + "/V*/" ) #moved annotated into the function
    
for key in videos:
    for video in videos[key]:
        makeVideo(video)


