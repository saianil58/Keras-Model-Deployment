# Import libraries
from flask import Flask, request, render_template  # handling Client & Server
from PIL import Image  # image handling
import re  # to find the raw data in image
import base64  # to decode the image data
from tensorflow.keras.models import load_model  # load the model into server
import numpy as np  # to convert datatype of image tensor

# Load the model
model = load_model(r'C:\ANN_Model.h5')
app = Flask(__name__)

# decoding an image from base64 into raw representation
def convertImage(imgData1):
    '''
    This Function will convert the raw image 
    given by user on screen as a tensor
    in the format required by the model
    '''
    imgstr = re.search(b"base64,(.*)", imgData1).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))
    # read the image into memory
    x = Image.open("output.png").convert('L').resize((28, 28))
    # compute a bit-wise inversion so black becomes white and vice versa
    x = np.invert(x)
    # conver the datatype as float32 as we our model will only work with numbers
    x = x.astype(np.float32)
    # reshape the image as input compatible of the model
    x = x.reshape(1, 28, 28)
    return x

# the below index function gets called when there is "/" in URL
@app.route('/')
def index():
    '''
    render out pre-built HTML file right on the index page
    '''
    return render_template("index.html")

# the below predict function gets called when there is "/predict/" in URL
@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    ''' 
    whenever the predict method is called, we're going
    to input the user drawn character as an image into the model
    perform inference, and return the classification
    '''
    # get the raw data format of the image
    imgData = request.get_data()
    # convert the base64 image to raw format
    x = convertImage(imgData)
    # get the prediction
    out = model.predict(x)
    # print the full list of probabilities
    print(out)
    # get the max probability
    print(np.argmax(out, axis=1))
    # convert the response to a string
    response = np.array_str(np.argmax(out, axis=1))
    return response

# Define the port on which the page should be displayed for the user
if __name__ == '__main__':
    app.run(port=8111, debug=True)