import os
from flask import Flask, render_template, request, redirect, abort
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png','.jpeg']
app.config['UPLOAD_PATH'] = 'static/uploads'
model = load_model("model1.h5")
class_names = ['a glioma tumor', 'a meningioma tumor', 'no tumor', 'a pituitary tumor']


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        upload_path = os.path.join(app.config['UPLOAD_PATH'], filename)
        img = tf.keras.utils.load_img(
            upload_path, target_size=(180, 180)
            )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
    return render_template('index.html',prediction="This image most likely contains {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)),filepath=upload_path)

    
if __name__ == '__main__':
    app.run(debug=True)