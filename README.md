# Keras-Model-Deployment

Often there’s a need to abstract away your machine learning model details and just deploy or integrate it with easy to use API endpoints. For eg., We can provide a URL endpoint using which anyone can make a POST request and they would get a JSON response of what the model has inferred without having to worry about its technicalities.

In this project, we will create a Flask server to deploy our MSIST classification artificial neural network (ANN) built in Keras. We will then create a simple Flask server which will accept POST request and do some image preprocessing and return the predictions.

Server code is [Here](https://github.com/saianil58/Keras-Model-Deployment/blob/master/Server_File.py)

Model building and saving is [Here](https://github.com/saianil58/Keras-Model-Deployment/blob/master/Classification_MNIST_ANN_Keras.ipynb)

For the data input of image we are using CSS here but we can use any technology as long as we call the API on our server in specifed path and required format.

CSS source code is [Here](https://github.com/saianil58/Keras-Model-Deployment/tree/master/static)

After the server is launced and we can go to specified url, in this case as we are doing local deployment the URL would be "http://127.0.0.1:8111/"

The screen on that url would look like this:


