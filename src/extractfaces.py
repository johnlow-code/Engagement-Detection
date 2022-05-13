import os
import cv2
import numpy as np

# Define paths

prototxt_path =  r"src\face_detector\deploy.prototxt"
caffemodel_path = r"src\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
# Read the model
model = cv2.dnn.readNet(prototxt_path, caffemodel_path)

dataset_path = "src\datasetimage"
categories_list = os.listdir(dataset_path)
for categories in categories_list:
    categories_path = os.path.join(dataset_path,categories)
    img_list = os.listdir(categories_path)
    for img in img_list:
        img_path=os.path.join(categories_path,img)
        image = cv2.imread(img_path)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        model.setInput(blob)
        detections = model.forward()
        # Create frame around face
        for i in range(0, detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            confidence = detections[0, 0, i, 2]

                # If confidence > 0.5, show box around face
            if (confidence > 0.5):
                frame = image[startY:endY, startX:endX]
                print("Faces Detected. Writing....",img)
                output_path =os.path.join(r"src\faces",categories,img)
                cv2.imwrite(output_path, frame)


print("Done!")
        # cv2.imwrite(base_dir + 'updated_images/' + file, image)
        # print("Image " + file + " converted successfully")
