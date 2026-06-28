


import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import datetime
from threading import Thread
import time
import pandas as pd

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise FileNotFoundError("Failed to load haarcascade_frontalface_default.xml")

ds_factor = 0.6

# Load emotion model
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0:"Angry", 1:"Disgusted", 2:"Fearful", 3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprised"}
music_dist = {
    0: "songs/angry.csv",
    1: "songs/disgusted.csv",
    2: "songs/fearful.csv",
    3: "songs/happy.csv",
    4: "songs/neutral.csv",
    5: "songs/sad.csv",
    6: "songs/surprised.csv"
}
global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1 
show_text = [0]

class WebcamVideoStream:
    def __init__(self, src=0):
        print(f"Initializing webcam with source: {src}")
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.stream.isOpened():
            raise RuntimeError(f"Failed to open webcam with source {src}")
        (self.grabbed, self.frame) = self.stream.read()
        if not self.grabbed:
            raise RuntimeError("Failed to read initial frame from webcam")
        self.stopped = False
    def start(self):
        print("Starting webcam thread")
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                print("Warning: Failed to grab frame")
    def read(self):
        return self.frame
    def stop(self):
        print("Stopping webcam")
        self.stopped = True
        self.stream.release()

def gen_frames():
    global cap1
    global df1
    try:
        cap1 = WebcamVideoStream(src=0).start()
        while True:
            image = cap1.read()
            print("Frame read from webcam")
            if image is None:
                print("Error: No frame captured")
                continue
            image = cv2.resize(image, (600, 500))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
            csv_path = music_dist[show_text[0]]
            try:
                df1 = pd.read_csv(csv_path)
                print(f"Loading CSV: {csv_path}")
                print("CSV columns:", df1.columns.tolist())
                if 'Unnamed: 0' in df1.columns:
                    df1 = df1.drop('Unnamed: 0', axis=1)
                df1 = df1.rename(columns={
                    'track_name': 'Name', 'name': 'Name',
                    'album': 'Album', 'album_name': 'Album',
                    'artist': 'Artist', 'artists': 'Artist',
                    'track_uri': 'URI', 'uri': 'URI', 'id': 'URI'
                })
                required_columns = ['Name', 'Album', 'Artist', 'URI']
                if 'URI' not in df1.columns:
                    print(f"Warning: 'URI' column missing in {csv_path}. Adding empty URI column.")
                    df1['URI'] = ''
                df1 = df1[required_columns].head(15)
            except Exception as e:
                print(f"Error loading CSV {csv_path}: {str(e)}")
                df1 = pd.DataFrame(columns=['Name', 'Album', 'Artist', 'URI'])
            for (x, y, w, h) in face_rects:
                cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
                roi_gray_frame = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                show_text[0] = maxindex 
                cv2.putText(image, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                df1 = music_rec()
            global last_frame1
            last_frame1 = image.copy()
            ret, jpeg = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ret:
                print("Error: Failed to encode JPEG")
                continue
            print("Yielding JPEG frame")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    except Exception as e:
        print(f"Error in gen_frames: {str(e)}")
    finally:
        if cap1:
            cap1.stop()

def music_rec():
    csv_path = music_dist[show_text[0]]
    try:
        df = pd.read_csv(csv_path)
        print(f"Loading CSV in music_rec: {csv_path}")
        print("CSV columns in music_rec:", df.columns.tolist())
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        df = df.rename(columns={
            'track_name': 'Name', 'name': 'Name',
            'album': 'Album', 'album_name': 'Album',
            'artist': 'Artist', 'artists': 'Artist',
            'track_uri': 'URI', 'uri': 'URI', 'id': 'URI'
        })
        required_columns = ['Name', 'Album', 'Artist', 'URI']
        if 'URI' not in df.columns:
            print(f"Warning: 'URI' column missing in {csv_path}. Adding empty URI column.")
            df['URI'] = ''
        df = df[required_columns].head(15)
    except Exception as e:
        print(f"Error in music_rec() for {csv_path}: {str(e)}")
        df = pd.DataFrame(columns=['Name', 'Album', 'Artist', 'URI'])
    return df