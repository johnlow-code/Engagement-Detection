import os
import cv2

data_path = "src\dataset"
folder_list = os.listdir(data_path)
print(folder_list)

for video in folder_list:
    video_path = os.path.join(data_path,video)  
    clipfolder_list = os.listdir(video_path)          
    for clipfolder in clipfolder_list:                
        clipfolder_path = os.path.join(video_path,clipfolder)     
        clip_list = os.listdir(clipfolder_path)
        for clip in clip_list:
            clip_path = os.path.join(clipfolder_path,clip) 
        # actual path would be src\dataset\*\*\*.avi
            print(clip_path)
            vid = cv2.VideoCapture(clip_path)
            while(True):
                ret, frame = vid.read()
                img_name = clipfolder+".jpg"
                img_path  = os.path.join("src\datasetimage",img_name)
                print("Creating....",img_name)
                cv2.imwrite(img_path,frame)
                break

