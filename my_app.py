# Import libraries
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
import pickle
import os
import re
import base64
os.chdir(r'C:\Users\samu0315\Desktop\Mine\Personal\gl_aiml\9.Neural Networks\Keras_Model_Deployment')
# Load the model
model = load_model('my_model.h5')
app = Flask(__name__)
# decoding an image from base64 into raw representation
def convertImage(imgData1):
    imgstr = re.search(b"base64,(.*)", imgData1).group(1)
    ##print(imgstr.dtype)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))
        
@app.route('/')
def index():
    # initModel()
    # render out pre-built HTML file right on the index page
    return render_template("index.html")
@app.route('/predict/',methods=['GET', 'POST'])
def predict():
    # whenever the predict method is called, we're going
    # to input the user drawn character as an image into the model
    # perform inference, and return the classification
    # get the raw data format of the image
    imgData = request.get_data()
    #imgData=imgData.decode('utf-8')
    # encode it into a suitable format
    #print('b4 conversion')
    convertImage(imgData)
    #print('after conversion')
    # read the image into memory
    x = Image.open("output.png").convert('L').resize((28,28))
    # compute a bit-wise inversion so black becomes white and vice versa
    x = np.invert(x)
    x = x.astype(np.float32)
    #print(x[0])
    # convert to a 4D tensor to feed into our model
    x = x.reshape(1,28,28 )
    #print('am here')
    out = model.predict(x)
    print(out)
    print(np.argmax(out, axis=1))
    # convert the response to a string
    response = np.array_str(np.argmax(out, axis=1))
    return response


if __name__ == '__main__':
    app.run(port=8111, debug=True)
