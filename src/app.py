#import asyncio
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
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from aiortc.contrib.media import MediaPlayer
from tensorflow.keras.applications import imagenet_utils				# Preprocess for any models
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


# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
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
    VGG19_URL = "https://www.dropbox.com/s/jm6qxnxhhafjfkq/VGG19.h5?dl=1"
    #VGG19_URL = "https://github.com/johnlow-code/Engagement-Detection/blob/main/src/models/VGG19.h5?raw=true"
    VGG19_LOCAL_PATH = HERE / "./models/VGG19.h5"

    download_file(MODEL_URL, MODEL_LOCAL_PATH,expected_size=10666211)
    download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH,expected_size=28092) 
    download_file(VGG19_URL, VGG19_LOCAL_PATH, expected_size=119153144)

    st.header("Engagement Detection")

    pages = {
        "Real time Detection (webcam)": app_real_time_detection,
        "Detect Engagement (photo)": app_image_detection,
        "Detect Engagement (video)": app_video_detection,
        #"Real time video transform with simple OpenCV filters (sendrecv)": app_video_filters,  # noqa: E501
        #"Delayed echo (sendrecv)": app_delayed_echo,
        #"Consuming media files on server-side and streaming it to browser (recvonly)": app_streaming,  # noqa: E501
        #"WebRTC is sendonly and images are shown via st.image() (sendonly)": app_sendonly_video,  # noqa: E501
        #"Simple video and audio loopback (sendrecv)": app_loopback,
        #"Configure media constraints and HTML element styles with loopback (sendrecv)": app_media_constraints,  # noqa: E501
        #"Control the playing state programatically": app_programatically_play,
        "Customize UI texts": app_customize_ui_texts
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


def app_real_time_detection():
    MODEL_LOCAL_PATH = HERE / "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    PROTOTXT_LOCAL_PATH = HERE / "./face_detector/deploy.prototxt"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    class Detection(NamedTuple):
        name: str
        prob: float

    class MobileNetSSDVideoProcessor(VideoProcessorBase):
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            self._net = cv2.dnn.readNet(
                str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
            )

            model_name = "VGG19.h5"
            model_path = os.path.join("src\models", model_name)
            self.engageNet = load_model('src/models/VGG19.h5')

            self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
            self.result_queue = queue.Queue()

        def _annotate_image(self, frame, locs, preds):
            # loop over the detections
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (disengaged, engaged) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                if engaged > 0.001:                             # Tune the Sensitivity here, default is if engaged > disengaged
                    label = "Engaged"
                    #engagedcount = engagedcount+1
                    color = (0, 255, 0)

                else:
                    label = "Disengaged"
                    #disengagedcount = disengagedcount+1
                    color = (0, 0, 255)
                
                #st.session_state.engagedcount = engagedcount
                #st.session_state.disengagedcount = disengagedcount
                #st.session_state.totalcount = engagedcount+disengagedcount
               
                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(disengaged, engaged) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            return frame

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            frame = frame.to_ndarray(format="bgr24")    # Change this if theres any problem with image format in bgr
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),    #Change to 224 for imagenet, 299 for xception, inception
		    (104.0, 177.0, 123.0))
            self._net.setInput(blob)
            detections = self._net.forward()
            print(detections.shape)

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
                    #face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))                 #Change here too
                    face = img_to_array(face)
                    face = imagenet_utils.preprocess_input(face)   #add imagenet_utils for models other than xception, inception

                    faces.append(face)
                    locs.append((startX, startY, endX, endY))

            if len(faces) > 0:
                faces = np.array(faces, dtype="float32")
                preds = self.engageNet.predict(faces, batch_size=32)

            annotated_image = self._annotate_image(frame, locs, preds)

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

    # confidence_threshold = st.slider(
    #     "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    # )
    # if webrtc_ctx.video_processor:
    #     webrtc_ctx.video_processor.confidence_threshold = confidence_threshold

    # if st.checkbox("Show the detected labels", value=True):
    #     if webrtc_ctx.state.playing:
    #         labels_placeholder = st.empty()
    #         # NOTE: The video transformation with object detection and
    #         # this loop displaying the result labels are running
    #         # in different threads asynchronously.
    #         # Then the rendered video frames and the labels displayed here
    #         # are not strictly synchronized.
    #         while True:
    #             if webrtc_ctx.video_processor:
    #                 try:
    #                     result = webrtc_ctx.video_processor.result_queue.get(
    #                         timeout=1.0
    #                     )
    #                 except queue.Empty:
    #                     result = None
    #                 labels_placeholder.table(result)
    #             else:
    #                 break

    # st.markdown()
    

def app_image_detection():
    MODEL_LOCAL_PATH = HERE / "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    PROTOTXT_LOCAL_PATH = HERE / "./face_detector/deploy.prototxt"
    VGG19_LOCAL_PATH = HERE / "./models/VGG19.h5"
    model_name = "VGG19.h5"
    model_path = os.path.join("src\models", model_name)
    engageNet = load_model(VGG19_LOCAL_PATH)
    faceNet  = cv2.dnn.readNet(
                str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
            )
    def detect_engagement(frame, faceNet, engageNet): 	#recv
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
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
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = imagenet_utils.preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = engageNet.predict(faces, batch_size=32)

        return (locs, preds)

    def predictimage(uploadedimage):
        img1 = Image.open(uploadedimage)
        frame = np.array(img1)
        # frame = cv2.imdecode(np.fromstring(uploadedimage.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            label = "Engaged" if engaged > 0.001 else "Disengaged"       # Tune the Sensitivity here, default is if engaged > disengaged
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

    uploadedimage = st.file_uploader("Upload an image", type=['jpeg','png','jpg','gif'], accept_multiple_files=False)
    if uploadedimage != None:
        predictimage(uploadedimage)
    

def app_video_detection():
    MODEL_LOCAL_PATH = HERE / "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    PROTOTXT_LOCAL_PATH = HERE / "./face_detector/deploy.prototxt"
    model_name = "VGG19.h5"
    model_path = os.path.join("src\models", model_name)
    engageNet = load_model('src/models/VGG19.h5')
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

    def detect_engagement(frame, faceNet, engageNet): 	#recv
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
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
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = imagenet_utils.preprocess_input(face)

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
        
        engagedcount=0
        disengagedcount=0
        st_video = open(vid,'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")
        cap = cv2.VideoCapture(vid)
        count = 0
        while True:
            _, frame = cap.read()
            if _ != False:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                    if engaged > 0.001:                      # Tune the Sensitivity here, default is if engaged > disengaged
                        label = "Engaged"
                        engagedcount = engagedcount+1
                        color = (0, 255, 0)

                    else:
                        label = "Disengaged"
                        disengagedcount = disengagedcount+1
                        color = (0, 0, 255)
                    
                    # include the probability in the label
                    label = "{}: {:.2f}%".format(label, max(disengaged, engaged) * 100)

                    # display the label and bounding box rectangle on the output
                    # frame
                    cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            else:
                break

        engagementrate = calc_engagementrate(engagedcount,disengagedcount)
        if engagedcount==0 or disengagedcount==0:
            st.warning("No face has been detected. Please try again with another video.")
        else:
            st.write("Engaged: ",engagementrate[0],"%, Disengaged:",engagementrate[1],"%")
        cap.release()
        #cv2.destroyAllWindows()

    uploadedvideo = st.file_uploader("Upload a video", type=['mp4','mpeg','mov'], accept_multiple_files=False)
    if uploadedvideo != None:
        predictvideo(uploadedvideo)


def app_customize_ui_texts():
    webrtc_streamer(
        key="custom_ui_texts",
        rtc_configuration=RTC_CONFIGURATION,
        translations={
            "start": "Mula",
            "stop": "Henti",
            "select_device": "Pilih peranti",
            "media_api_not_available": "API media tiada",
            "device_ask_permission": "Peranti minta kebenaran",
            "device_not_available": "Peranti tidak sedia",
            "device_access_denied": "Akses peranti telah dinafikan",
        },
    )


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
