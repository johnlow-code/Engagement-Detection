import asyncio
import datetime
import pandas as pd
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
from aiortc.contrib.media import MediaPlayer
#from tensorflow.keras.applications.vgg19 import preprocess_input				# Preprocess for any models
from tensorflow.keras.applications.inception_v3 import preprocess_input	# Preprocess for Inception and Xception
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from PIL import Image
import imutils

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]} 
)


def main():
    MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"  # noqa: E501
    MODEL_LOCAL_PATH = HERE / "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    PROTOTXT_URL = "https://github.com/jahnavi-prasad/face-mask-detection/raw/master/face_detector/deploy.prototxt"  # noqa: E501
    PROTOTXT_LOCAL_PATH = HERE / "./face_detector/deploy.prototxt"
    #VGG19_URL = "https://www.dropbox.com/s/jm6qxnxhhafjfkq/VGG19.h5?dl=1"
    #VGG19_URL = "https://github.com/johnlow-code/Engagement-Detection/blob/main/src/models/VGG19.h5?raw=true"
    #VGG19_LOCAL_PATH = HERE / "./models/VGG19.h5"
    InceptionV3_URL="https://www.dropbox.com/s/76wezvqe70ore9e/InceptionV3.h5?dl=1"
    InceptionV3_LOCAL_PATH = HERE / "./models/InceptionV3.h5"
    

    download_file(MODEL_URL, MODEL_LOCAL_PATH,expected_size=10666211)
    download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH,expected_size=28092) 
    download_file(InceptionV3_URL, InceptionV3_LOCAL_PATH, expected_size=289840008)

    st.header("Engagement Detection")

    pages = {
        "Real time Detection (webcam)": app_real_time_detection,
        "Detect Engagement (photo)": app_image_detection,
        "Detect Engagement (video)": app_video_detection,
        "User Manual Page": app_user_manual
        #"Customize UI texts": app_customize_ui_texts
    }
    page_function = pages.keys()

    selected_page = st.sidebar.selectbox(
        "Choose the app mode",
        page_function
    )
    st.subheader(selected_page)

    pages[selected_page]()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


def app_loopback():
    """Simple video loopback"""
    webrtc_streamer(key="loopback")

def calc_engagement(engagedcount,disengagedcount):
    total = engagedcount+disengagedcount
    if total == 0:
        return None
    return (engagedcount/total*100.0)

def app_real_time_detection():
    #global enagagedcount
    #engagedcount = [0,0]
    #df = pd.DataFrame(columns=['Minutes','Engagement Rate'])
    #ratedata = np.array([])
    #rate = 0
    #num_minutes = 0
    MODEL_LOCAL_PATH = HERE / "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    PROTOTXT_LOCAL_PATH = HERE / "./face_detector/deploy.prototxt"
    InceptionV3_LOCAL_PATH = HERE / "./models/InceptionV3.h5"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    #class Detection(Named):
    #    engagestatus: str

    class MobileNetSSDVideoProcessor(VideoProcessorBase):
        result_queue: "queue.Queue[List]"

        def __init__(self) -> None:
            self._net = cv2.dnn.readNet(
                str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
            )

            model_name = "InceptionV3.h5"
            self.engageNet = load_model(InceptionV3_LOCAL_PATH)

            self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
            self.result_queue = queue.Queue()

        def _annotate_image(self, frame, locs, preds):
            result: List = []
            # loop over the detections
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (disengaged, engaged) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                if engaged > self.confidence_threshold:                             # Tune the Sensitivity here, default is if engaged > disengaged
                    label = "Engaged"
                    color = (0, 255, 0)

                else:
                    label = "Disengaged"
                    color = (0, 0, 255)
                
                result.append(label)
                #st.session_state.engagedcount = engagedcount
                #st.session_state.disengagedcount = disengagedcount
                #st.session_state.totalcount = engagedcount+disengagedcount
                #yield engagedcount
                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(disengaged, engaged) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            return frame, result

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            frame = frame.to_ndarray(format="bgr24")    # Change this if theres any problem with image format in bgr
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (299,299),    #Change to 224 for imagenet, 299 for xception, inception
		    (104.0, 177.0, 123.0))
            self._net.setInput(blob)
            detections = self._net.forward()
            #print(detections.shape)

            faces = []
            locs = []
            preds = []
            
            for i in range(0, detections.shape[2]):

                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    face = frame[startY:endY, startX:endX]
                    face = cv2.resize(face, (299, 299))                 #Change here too
                    face = img_to_array(face)
                    face = preprocess_input(face)   #add imagenet_utils for models other than xception, inception

                    faces.append(face)
                    locs.append((startX, startY, endX, endY))

            if len(faces) > 0:
                faces = np.array(faces, dtype="float32")
                preds = self.engageNet.predict(faces, batch_size=32)

            annotated_image, result = self._annotate_image(frame, locs, preds)
            if result != []:
                self.result_queue.put(result)

            # NOTE: This `recv` method is called in another thread,
            # so it must be thread-safe.
            #self.result_queue.put(result)

            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")  # Change this if theres any problem with the video format in bgr

    webrtc_ctx = webrtc_streamer(
        key="real-time-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=MobileNetSSDVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    confidence_threshold = st.slider(
       "Sensitivity to engagement", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence_threshold = 1-confidence_threshold

    if st.checkbox("Show Stats", value=True):
        if webrtc_ctx.state.playing:
            ratedata = []
            max_runtime = 1             #Average engagement rate per X seconds
            # NOTE: The video transformation with object detection and
            # this loop displaying the result labels are running
            # in different threads asynchronously.
            # Then the rendered video frames and the labels displayed here
            # are not strictly synchronized.
            while True:
                if webrtc_ctx.input_video_track.readyState=="live":
                    start_time = datetime.datetime.now()
                    engagedarray = []
                    while (datetime.datetime.now()-start_time).seconds < max_runtime:
                        try:
                            result = webrtc_ctx.video_processor.result_queue.get(
                                timeout=1.0
                            )
                            #append it to an array
                            engagedarray.append(result)
                        except queue.Empty:
                            result = None
                        
                    ratedata.append(calc_engagement(engagedarray.count(['Engaged']),engagedarray.count(['Disengaged'])))
                    #engagedarray.clear()
                    start_time = datetime.datetime.now()
                else:
                    if(sum(1 for x in ratedata if x != None)==0):
                        if "dataframe" in st.session_state:
                            del st.session_state["dataframe"]
                        st.session_state.warning=True
                        break
                    df = pd.DataFrame(columns=['Minutes','Engagement Rate'])
                    df = df.append(pd.DataFrame(ratedata, columns=['Engagement Rate']), ignore_index=True)
                    x = list(range(1,len(ratedata)+1))
                    df = df.append(pd.DataFrame(x, columns=['Minutes']), ignore_index=True)
                    st.session_state.dataframe = df
                    average_rate = sum(filter(None, ratedata))/sum(1 for x in ratedata if x != None)
                    st.session_state.average_rate=average_rate
                    break
        
        if "warning" in st.session_state:
            st.warning('"Hello! Are you there?" Stats are unavailable as no face is detected. :P')
            del st.session_state["warning"]
        if "dataframe" in st.session_state:
            st.write("Average Engagement Rate: {:0.2f}%".format(st.session_state["average_rate"]))
            df = st.session_state["dataframe"]
            st.line_chart(df["Engagement Rate"])
        

            

        #plot graph 

    # st.markdown()
    

def app_image_detection():
    MODEL_LOCAL_PATH = HERE / "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    PROTOTXT_LOCAL_PATH = HERE / "./face_detector/deploy.prototxt"
    InceptionV3_LOCAL_PATH = HERE / "./models/InceptionV3.h5"
    model_name = "InceptionV3.h5"
    engageNet = load_model(InceptionV3_LOCAL_PATH)
    faceNet  = cv2.dnn.readNet(
                str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
            )
    def detect_engagement(frame, faceNet, engageNet): 	#recv
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (299, 299),
            (104.0, 177.0, 123.0))

        faceNet.setInput(blob)
        detections = faceNet.forward()
        print(detections.shape)

        faces = []
        locs = []
        centroids = []
        preds = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                face = cv2.resize(face, (299, 299))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = engageNet.predict(faces, batch_size=32)

        return (locs, preds)

    def predictimage(uploadedimage):
        with st.spinner('Please wait...'):
            img1 = Image.open(uploadedimage)
            frame = np.array(img1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = imutils.resize(frame, width=400)
            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = detect_engagement(frame, faceNet, engageNet)
            # loop over the detected face locations and their corresponding
            # locations
        st.subheader("Result: ")

        if locs==[]:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictedimage = st.image(frame)
            st.warning("No face has been detected. Please try again with another image.")
            return

        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (disengaged, engaged) = pred
            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Engaged" if engaged > disengaged else "Disengaged"       # Tune the Sensitivity here, default is if engaged > disengaged
            st.title(label)
            color = (0, 255, 0) if label == "Engaged" else (0, 0, 255)
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(disengaged, engaged) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            # show the output frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictedimage = st.image(frame)
        st.text(label)
        #cv2.destroyAllWindows()

    uploadedimage = st.file_uploader("Upload an image", type=['jpeg','png','jpg','gif','heic'], accept_multiple_files=False)
    if uploadedimage != None:
        predictimage(uploadedimage)
    

def app_video_detection():
    MODEL_LOCAL_PATH = HERE / "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    PROTOTXT_LOCAL_PATH = HERE / "./face_detector/deploy.prototxt"
    InceptionV3_LOCAL_PATH = HERE / "./models/InceptionV3.h5"
    model_name = "InceptionV3.h5"
    engageNet = load_model(InceptionV3_LOCAL_PATH)
    faceNet  = cv2.dnn.readNet(
                str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
            )
    def calc_engagementrate(engagedcount, disengagedcount):
        if engagedcount+disengagedcount==0:
            return ""
        total = engagedcount+disengagedcount
        engagedrate = (float(engagedcount)/float(total))*100
        engagedrate = round(engagedrate, 2)
        disengagedrate = 100 - engagedrate
        return [engagedrate, disengagedrate]

    def calc_engagementrateonly(engagedcount, disengagedcount):
        if engagedcount+disengagedcount==0:
            return 0
        total = engagedcount+disengagedcount
        engagedrate = (float(engagedcount)/float(total))*100
        engagedrate = round(engagedrate, 2)
        return engagedrate

    def detect_engagement(frame, faceNet, engageNet): 	#recv
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (299, 299),
            (104.0, 177.0, 123.0))

        faceNet.setInput(blob)
        detections = faceNet.forward()
        #print(detections.shape)

        faces = []
        locs = []
        centroids = []
        preds = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                face = cv2.resize(face, (299, 299))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = engageNet.predict(faces, batch_size=32)

        return (locs, preds)

    def predictvideo(uploadedvideo):
        vid = uploadedvideo.name
        with open(vid, mode='wb') as f:
            f.write(uploadedvideo.read()) #save the video to disk
        
        max_runtime = 1

        engagecount= [0,0]
        engagecount_temp = [0,0]
        st_video = open(vid,'rb')
        video_bytes = st_video.read()
        processing_text = st.empty()
        processing_text.write("Processing video...")
        cap = cv2.VideoCapture(vid)
        ratedata_video = np.array([])
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fps = fps*max_runtime
        frame_count=0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        steps = (100.0/total)/100
        progress = 0.0 - steps
        progressbar = st.progress(0)
        frame_window = st.image([])
        while True:
            _, frame = cap.read()

            if _ != False:
                progress = progress+steps
                progressbar.progress(progress)
                frame = imutils.resize(frame, width=400)

                (locs, preds) = detect_engagement(frame, faceNet, engageNet)
                # loop over the detected face locations and their corresponding
                # locations
                for (box, pred) in zip(locs, preds):
                    # unpack the bounding box and predictions
                    (startX, startY, endX, endY) = box
                    (disengaged, engaged) = pred

                    # determine the class label and color we'll use to draw
                    # the bounding box and text
                    if engaged > disengaged:                      # Tune the Sensitivity here, default is if engaged > disengaged
                        label = "Engaged"
                        engagecount_temp[1] = engagecount_temp[1]+1
                        color = (0, 255, 0)

                    else:
                        label = "Disengaged"
                        engagecount_temp[0] = engagecount_temp[0]+1
                        color = (0, 0, 255)
                    
                    # include the probability in the label
                    label = "{}: {:.2f}%".format(label, max(disengaged, engaged) * 100)

                    # display the label and bounding box rectangle on the output
                    # frame
                    cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
                if(frame_count%fps==0):
                    ratedata_video = np.append(ratedata_video, calc_engagementrateonly(engagecount_temp[1],engagecount_temp[0]))
                    engagecount[0]=engagecount_temp[0] + engagecount[0]
                    engagecount[1]=engagecount_temp[1]+engagecount[1]
                    engagecount_temp=[0,0]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_window.image(frame)
                frame_count = frame_count+1

            else:
                break

        frame_window.empty()
        progressbar.empty()
        processing_text.empty()
        
        if sum(engagecount)==0:
            st.warning("No face has been detected. Please try again with another video.")
        else:
            st.subheader("Result: ")
            df_video = pd.DataFrame(columns=['Minutes','Engagement Rate'])
            df_video = df_video.append(pd.DataFrame(ratedata_video, columns=['Engagement Rate']), ignore_index=True)
            x = list(range(1,len(ratedata_video)+1))
            df_video = df_video.append(pd.DataFrame(x, columns=['Minutes']), ignore_index=True)
            engagementrate = calc_engagementrateonly(engagecount[1],engagecount[0])
            st.write("Average Engagement Rate: {:0.2f}%".format(engagementrate))
            st.line_chart(df_video["Engagement Rate"])
            #frame_window.image(frame)
            
        cap.release()
        uploadedvideo = None

    uploadedvideo = st.file_uploader("Upload a video", type=['mp4','mpeg','mov','avi','wmv'], accept_multiple_files=False)
    if uploadedvideo != None:
        predictvideo(uploadedvideo)

def app_user_manual():
    st.info("Welcome! This app is developed by Low Jun Jie for the Final Year Project. Hope this user manual can guide you on using the app.")
    st.subheader("Navigating the website")
    st.image("src/media/usm-vid8.gif")
    st.markdown("You can navigate the website and switch between application modes via the sidebar as shown above.")
    st.subheader("Real time Detection (webcam)")
    st.image("src/media/usm2.PNG")
    st.markdown("First, click on *SELECT DEVICE*")
    st.image("src/media/usm3.PNG")
    st.markdown("Your browser should show a pop-up prompt stating that streamlit.io wants to use your camera. Click allow.")
    st.image("src/media/usm4.png")
    st.markdown("You will then be able to choose which camera input device you want to use (provided if you have more than one camera, such as a front camera and back camera on a smartphone.)")
    st.image("src/media/usm1.png")
    st.markdown("Click *START* and you're ready to go! If nothing shows up, please click *STOP* and *START* button again.")
    st.image("src/media/usm5.png")
    st.markdown("If you would like the app to track the rate of engagement, make sure that the *Show Stats* checkbox is ticked.")
    st.image("src/media/usm-vid1.gif")
    st.markdown("Sometimes, the app will have a tendency of falsely detecting the user as disengaged even when the user is actually engaged. To fix this, you can adjust the *Sensitivity to Engagement* slider.")
    st.markdown("Make sure to restart the camera video stream(as shown above) in order for any changes to take effect.")
    st.image("src/media/usm6.png")
    st.markdown("If **NotReadableError** occured, make sure that your camera isn't currently being used for another application.")
    st.subheader("Engagement Rate Graph")
    st.markdown("When a face is detected in the webcam or video app mode, an engagement rate graph will be shown. ")
    st.image("src/media/usm-vid6.gif")
    st.markdown("You can zoom in/out and pan the graph however you like.")
    st.image("src/media/usm-vid3.gif")
    st.markdown("You can also download the graph by accessing the three dots menu.")
    st.image("src/media/usm-vid7.gif")
    st.markdown("You can expand the graph to be shown fullscreen too!")
    st.subheader("Detect Engagement (Photo)")
    st.image("src/media/usm-vid4.gif")
    st.markdown("You can upload an image to have the app predict whether the face detected is disengaged or engaged.")
    st.image("src/media/usm-vid5.gif")
    st.markdown("If you would like to upload another image, simply click the *Browse files* button again.")
    st.subheader("Detect Engagement (Video)")
    st.image("src/media/usm-vid2.gif")
    st.markdown("You can upload a video to detect engagement. Once the video is uploaded, you will see a progress bar along with some preview of the prediction results.")
    st.markdown("Once the processing is done, you will be greeted with an average engagement rate result and a graph containing the engagement rate throughout the length of the video.")
    st.subheader("Accessibility - Changing the color theme of the app")
    st.image("src/media/usm-vid9.gif")
    st.markdown("You can change between light mode / dark mode as shown above.")
    st.subheader("Debugging")
    st.image("src/media/usm-vid10.gif")
    st.markdown("Rerun is a useful feature to have if you ever encounter any bugs or if the application somehow stops working.")
    st.image("src/media/usm6.png")
    st.markdown("If **NotReadableError** occured in the real time detection app, make sure that your camera isn't currently being used on another application.")
    st.markdown("The app is currently being hosted on share.streamlit.io on a free plan, which means the application may run out of its limited bandwidth/memory.")
    st.markdown("Please do not hesitate to contact Jun Jie if this happens. Thank you for your support and suggestions for improvements!")


# def app_customize_ui_texts():
#     webrtc_streamer(
#         key="custom_ui_texts",
#         rtc_configuration=RTC_CONFIGURATION,
#         translations={
#             "start": "Mula",
#             "stop": "Henti",
#             "select_device": "Pilih peranti",
#             "media_api_not_available": "API media tiada",
#             "device_ask_permission": "Peranti minta kebenaran",
#             "device_not_available": "Peranti tidak sedia",
#             "device_access_denied": "Akses peranti telah dinafikan",
#         },
#     )


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
