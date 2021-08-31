import cv2
from flask import Flask, render_template, redirect
import numpy as np
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
  
# Flask Setup
app = Flask(__name__)

model_path = os.path.join('static','models','emotion_model_full.h5')

def load_model(model_path):
    emotion_model = tf.keras.models.load_model(model_path)
    
    return emotion_model
    
emotion_model = load_model(model_path) 
# Flask Routes
@app.route("/")
def index():
    return render_template("indexmer.html")

    
@app.route("/camera")
def camera():
    
    def emotion_prediction(emotion_model):
        cv2.ocl.setUseOpenCL(False)
        emotion_dict = {0: "Angry",
                        1: "Disgusted",
                        2: "Fearful",
                        3: "Happy",
                        4: "Neutral",
                        5: "Sad",
                        6: "Surprised"}
        bounding_box_path = os.path.join('static','xml','haarcascade_frontalface_default.xml')
        # start the webcam feed
        try:
            cap = cv2.VideoCapture(0)
            while True:
                # Use haar cascade to draw bounding box around face
                ret, frame = cap.read()
                if not ret:
                    break
                bounding_box = cv2.CascadeClassifier(bounding_box_path)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in num_faces:
                    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                    roi_gray_frame = gray_frame[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                    emotion_prediction = emotion_model.predict(cropped_img)
                    maxindex = int(np.argmax(emotion_prediction))
                    cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    print(maxindex)

                cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except (KeyboardInterrupt, Exception) as e:
            cap.release()
            cv2.destroyAllWindows()
            print(repr(e))

        cap.release()

    emotion_prediction(emotion_model)
    

    
    

    

if __name__ == "__main__":
	app.run(debug=True)