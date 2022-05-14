# -*- coding: utf-8 -*-
"""
@author: John Low
"""
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from imutils import paths
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

INIT_LR = 1e-4  # Initial Learning Rate     Control learning rate, increase 
EPOCHS = 50    # How many runs of trainings
BS = 64         # Batch Size of the training
DIRECTORY = r"src\faces"
CATEGORIES = ["0", "1"]
img_shape = (224,224,3)
preprocess = imagenet_utils.preprocess_input

model_num = int(input("Select a CNN Architecture:\n1.VGG19 \n2.DenseNet \n3.ResNet \n4.Inception \n5.Xception \n"))
while model_num not in range(1,6):
    print("\nPlease enter a number between 1-5.\n")
    model_num = int(input("Select a CNN Architecture:\n1.VGG19 \n2.DenseNet \n3.ResNet \n4.Inception \n5.Xception \n"))

model_name = {1:VGG19, 2:DenseNet121, 3:ResNet50V2, 4:InceptionV3, 5:Xception }
model_name_str = {1:"VGG19", 2:"DenseNet121", 3:"ResNet50V2", 4:"InceptionV3", 5:"Xception" }

if model_num==4 or model_num==5:
    preprocess = preprocess_input
    img_shape = (299,299,3)

#  Images

print("[INFO] loading images")
data = []   #Image arrays are appended in it
labels = [] #Appends image labels

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)  #specific path of one img
        # LOAD IMAGE AND CONVERT SIZE TO 56x56m grayscale images
        image = load_img(img_path, target_size=img_shape)
        image = img_to_array(image) # KERAS: convert image to array
        image = preprocess(image) # scale input pixels between -1 to 1.
        
        # Append Image to data list
        data.append(image)
        
        # Appends labels to label list
        labels.append(category)
        

# Encoding the labels as 0 and 1
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
      test_size = 0.20, stratify=labels, random_state=53)
# Refer to the diagram in the report for the meaning of x train, y train etc

# Construct the training image generator for data augmentation
aug = ImageDataGenerator(
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    rotation_range=20,
    zoom_range=0.15, #Randomly zoom with tange of 0.15
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    # brightness_range=[1.0,1.25],
    fill_mode="nearest"
    )

# load the CNN Architecture here!
# imagenet is used as the predefined weights for images
# 1 for 1 grayscale channels
baseModel = model_name[model_num](weights="imagenet", include_top=False,       #Dont include top for transfer learning
                        input_tensor=Input(shape=img_shape))

# construct the head of the model that will be placed on top of the base mmodel
headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel) #non linear, for images
headModel = Dropout(0.3)(headModel) # Dropout rate, careful of overfitting
headModel = Dense(128, activation="relu")(headModel) #non linear, for images
headModel = Dense(128, activation="relu")(headModel) #non linear, for images
headModel = Dropout(0.3)(headModel) # Dropout rate, careful of overfitting
headModel = Dense(2, activation="softmax")(headModel) #number of categories

# place head FC model on top of base model
# This is the ACTUAL model we will train
model = Model(inputs=baseModel.input, outputs=headModel)
# print(model.summary())
# loop over all layers in the base model and freeze them so they will not
# be updated during the first training
for layer in baseModel.layers:
    layer.trainable = False

# Compile Mdoel
print("[INFO] Compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# Train the head of the NN
print("[INFO] Training head!")
aug.fit(trainX)
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX,testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)


   
# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the label with
# corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
report = classification_report(testY.argmax(axis=1), predIdxs, output_dict=True,
                            target_names=lb.classes_)
print(report)
results_path = os.path.join(r"src\results",str(model_name_str[model_num]))

df = pd.DataFrame(report).transpose()
df.to_csv(os.path.join(results_path,"report.csv"))
mat = confusion_matrix(testY.argmax(axis=1),predIdxs)
plot_confusion_matrix(conf_mat=mat)
plt.savefig(os.path.join(results_path,"ConfusionMatrix.png"))

# SERIALIZE model to the disk
print("[INFO] saving Engagement detection model...")
model_filename= str(model_name_str[model_num]) +".h5"                                  
model_path = os.path.join(r"src\models",model_filename)
model.save(model_path, save_format="h5")

# Plot training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(os.path.join(results_path,"plot.png"))




    
        
