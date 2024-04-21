import os
from flask import Flask, render_template, request, send_from_directory
from keras_preprocessing import image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

STATIC_FOLDER = 'static'
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'
MODEL_FOLDER = STATIC_FOLDER + '/models'

# Load model once at running time for all the predictions
print('[INFO] : Model loading ................')
model = tf.keras.models.load_model(MODEL_FOLDER + '/cat_dog_classifier.h5')
print('[INFO] : Model loaded')


def predict(fullpath):
    data = image.load_img(fullpath, target_size=(128, 128))
    data = image.img_to_array(data)
    data = np.expand_dims(data, axis=0)
    data = data.astype('float32') / 255

    result = model.predict(data)

    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        result = predict(fullname)

        pred_prob = result.item()

        if pred_prob > .5:
            label = 'Dog'
            accuracy = round(pred_prob * 100, 2)
        else:
            label = 'Cat'
            accuracy = round((1 - pred_prob) * 100, 2)

        return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
