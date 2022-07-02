#Importing all the libraries needed
import tensorflow as tf

from keras import  preprocessing, Input
#from multimodel baseline functions
from keras.utils.vis_utils import  plot_model
#import keras matrics

import os
import pandas as pd
from nltk.corpus import stopwords
# from nltk import word_tokenize
from keras.preprocessing import image
from keras.applications.vgg16 import  preprocess_input
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.layers import  multiply
from PIL import Image, ImageFile

try:
    from PIL import Image
except ImportError:
    import Image


model_load =tf.keras.models.load_model(r'model_VGG.h5',compile=False)
i="0SvkQMd.png"
# Flask utils
from flask import Flask,  request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer





# Define a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(128, 128,3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    print("prediction:",preds)
    a = preds
    
    ind=np.argmax(a)     
    return ind

@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
    
        # Get the file from post request
        print(request.files)
        f = request.files['image']
      
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        print("Base--",basepath)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print("Fileee----------------------------------")
        print("File----",file_path)
        img = image.load_img(file_path, grayscale=False, target_size=(128, 128))
        x = image.img_to_array(img)
        print(x)
        x = np.expand_dims(x, axis=0)
        x = np.array(x, 'float32')

        print(x.shape)
        # x /= 255
        # result=model_predict(file_path,model_load)
        preds=model_load.predict(x)
        print("prediction",preds)
        # print(result)
        print("Res",preds[0])
        pred=preds[0][0]
        print(pred)
        res={}
        if(preds==0):
            res['status']=0
        else:
            res['status']=1
        return res

if __name__ == '__main__':
   app.run(debug=True,port=5001)