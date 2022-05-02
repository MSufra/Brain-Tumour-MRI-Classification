import os
from flask import Flask, render_template, request, redirect, abort
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func, Column, Integer, String


app = Flask(__name__)

#DATABASE SETUP#
Base = declarative_base()

class Patient(Base):
    __tablename__ = 'patient'
    id = Column(Integer, primary_key=True)
    fname = Column(String(50))
    lname = Column(String(50))
    age = Column(Integer)
    sex = Column(String(10))
    pred= Column(String(50)) 

engine = create_engine("postgresql://postgres:postgres@localhost:5432/BTC")
conn = engine.connect()

#MODEL CONFIG#
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
        prediction = class_names[np.argmax(score)]
        fname = request.form.get('fname')
        lname = request.form.get('lname')
        age = request.form.get('age')
        sex = request.form.get('sex')
        patient = Patient(fname=fname,lname=lname,age=age,sex=sex,pred=prediction)
        session = Session(bind=engine)
        session.add(patient)
        try:
            session.commit()
        except:
            session.close()
            abort(400)    
        session.close()
    return render_template('index.html',prediction="This image most likely contains {} with a {:.2f} percent confidence.".format(prediction, 100 * np.max(score)),filepath=upload_path)

@app.route('/patients')
def patients():
    session = Session(bind=engine)
    rows = session.query(Patient)
    return render_template('patients.html',rows = rows)    

if __name__ == '__main__':
    app.run(debug=True)