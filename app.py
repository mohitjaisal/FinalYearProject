from flask import Flask,render_template,request
from tensorflow.keras.utils import load_img
from keras_preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
# from gevent.pywsgi import WSGIServer


app = Flask(__name__)
Model_Path= 'models/pneu_cnn_model.h5'
model = load_model(Model_Path)

@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/',methods=['POST','GET'])
def predict():
    imagefile= request.files["imagefile"]
    image_path ='./static/' + imagefile.filename
    imagefile.save(image_path)
    img=load_img(image_path,target_size=(500,500),color_mode='grayscale')
    x=img_to_array(img)
    x=x/255
    x=np.expand_dims(x, axis=0)
    classes=model.predict(x)
    result1=classes[0][0]
    result2='Negative'
    if result1>=0.5:
        result2='Positive'
    classification ='%s (%.2f%%)' %(result2,result1*100)
    return render_template('index.html',prediction=classification,imagePath=image_path)
if __name__ == '__main__':
    app.run(port=5000,debug=True)