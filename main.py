import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np


SIZE = 28

app = Flask(__name__)

model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/send', methods=['POST'])
def predict():
    file = request.files.get("file")
    text = 'ファイルが正しく選択されていません'
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join("uploads", filename))
        filepath = os.path.join("uploads", filename)
        img = image.load_img(filepath, grayscale=True, target_size=(SIZE, SIZE))
        img = image.img_to_array(img)
        data = np.array([img])
        result = model.predict(data)[0]
        pred = result.argmax()
        # img = Image.open(image).convert('L').resize((SIZE, SIZE))
        # pred = model.predict(np.array(img).reshape(1, SIZE, SIZE))
        text = '{}が予測されました'.format(np.argmax(pred))
    return render_template('index.html', text=text)

