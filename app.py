from flask import Flask, render_template, request, jsonify
import io
from tensorflow.keras.models import model_from_json
import cv2
import numpy as np
import base64
from PIL import Image
from tensorflow.keras.preprocessing import image

app = Flask(__name__,template_folder='templates')

model = model_from_json(open("fmm_model.json", "r").read())
model.load_weights('fmm_model_weights.h5')

emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

@app.route('/get_image', methods=['GET','POST'])
def get_image():
    imgURL = request.values['imgURL']
    imageb64 = imgURL.split(',')[1]
    # Take in base64 string and return PIL image
    def stringToImage(base64_string):
        imgdata = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(imgdata))

    # convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
    def toRGB(image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    img = toRGB(stringToImage(imageb64))

    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    prediction = []
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    for (x,y,w,h) in faces_detected:
            cv2.rectangle(gray_img,(x,y),(x+w,y+h),(255,255,255),thickness=2)    
            roi_gray=gray_img[y:y+w,x:x+h]  #cropping face area from  image  
            roi_gray=cv2.resize(roi_gray,(48,48)) 
            img_pixels = image.img_to_array(roi_gray)  
            img_pixels = np.expand_dims(img_pixels, axis = 0)  
            img_pixels /= 255
            prediction = model.predict(img_pixels)[0]
    try:
        pred_emotion = emotions[np.argmax(prediction)]
        score_val = prediction[np.argmax(prediction)]*100
        pred_emotion_on_img = str(round(score_val))+ '% ' +str(pred_emotion) 
        text_pos = (int(x-10), int(y-10))
    except(ValueError):
        pred_emotion_on_img = 'Try again'
        pred_emotion = 'Try again'
        score_val = 0
        text_pos = (100,100)

    cv2.putText(gray_img, pred_emotion_on_img, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    pil_result=Image.fromarray(gray_img)
    buffered = io.BytesIO()
    pil_result.save(buffered, format="PNG")
    send_img = base64.b64encode(buffered.getvalue()).decode()
    return jsonify({'score_val':str(round(score_val)),'pred_emotion':pred_emotion,'img':send_img})

@app.route('/')
def index():
    return render_template('index.html')
