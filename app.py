from flask import Flask
from flask.templating import render_template
from flask import request
import tensorflow.keras as keras
import librosa 
import math 
import numpy as np
import os
from werkzeug.utils import secure_filename
app = Flask(__name__)

folder = "/F:/samples/"

def process_input(audio_file, track_duration):

  SAMPLE_RATE = 22050
  NUM_MFCC = 13
  N_FTT=2048
  HOP_LENGTH=512
  TRACK_DURATION = track_duration # measured in seconds
  SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
  NUM_SEGMENTS = 10

  samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
  num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)

  signal, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)
  
  for d in range(10):

    # calculate start and finish sample for current segment
    start = samples_per_segment * d
    finish = start + samples_per_segment

    # extract mfcc
    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=NUM_MFCC, n_fft=N_FTT, hop_length=HOP_LENGTH)
    mfcc = mfcc.T

    return mfcc


@app.route('/')
def predict():
    return render_template("index.html")


@app.route('/getGenre',methods=['POST'])
def getGenre():
    genre_dict = {10:"blues",1:"metal",2:"country",0:"disco",4:"classical",5:"jazz",3:"rock",9:"pop",8:"hiphop",7:"reggae"}
    file = request.files['x']
    uploads_dir =""
    #os.makedirs(uploads_dir)
    file.save(os.path.join(uploads_dir, secure_filename(file.name)))
    mfccs = process_input(os.path.abspath(file.name), 30)
    X_to_predict = mfccs[np.newaxis, ..., np.newaxis]
    model = keras.models.load_model("Music_Genre_10_CNN.h5")
    prediction = model.predict(X_to_predict)
    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)
    genre = genre_dict[ int(predicted_index) ]
    return render_template("index.html", genre = genre)

if __name__ == "__main__":
    app.run(debug = True)