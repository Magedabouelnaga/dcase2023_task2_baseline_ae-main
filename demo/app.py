from flask import Flask, request, render_template, jsonify
import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf
import os

app = Flask(__name__)
model = tf.keras.models.load_model('model/my_model.h5')

def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    y = librosa.util.normalize(y)
    mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram_db = np.expand_dims(mel_spectrogram_db, axis=0)  # Add batch dimension
    mel_spectrogram_db = np.expand_dims(mel_spectrogram_db, axis=-1)  # Add channel dimension
    return mel_spectrogram_db

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)
        features = preprocess_audio(file_path)
        prediction = model.predict(features)
        os.remove(file_path)
        result = 'Abnormal' if prediction > 0.5 else 'Normal'
        return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
