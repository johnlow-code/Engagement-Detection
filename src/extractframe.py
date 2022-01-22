import os
import cv2

data_path = "src\dataset"
folder_list = os.listdir(data_path)
print(folder_list)

for video in folder_list:
    video_path = os.path.join(data_path,video)  #dataset/110001
    clipfolder_list = os.listdir(video_path)          #dataset/110001/*
    for clipfolder in clipfolder_list:                #clipfolder = 1100011002
        clipfolder_path = os.path.join(video_path,clipfolder)     #dataset/110001/1100011002
        clip_list = os.listdir(clipfolder_path)
        for clip in clip_list:
            clip_path = os.path.join(clipfolder_path,clip) #dataset/110001/1100011002/1100011002.avi 
        # actual path would be src\dataset\*\*\*.avi
        # extract frame code here! https://www.geeksforgeeks.org/extract-images-from-video-in-python/
            print(clip_path)
            vid = cv2.VideoCapture(clip_path)
            while(True):
                ret, frame = vid.read()
                img_name = clipfolder+".jpg"
                img_path  = os.path.join("src\datasetimage",img_name)
                print("Creating....",img_name)
                cv2.imwrite(img_path,frame)
                break

