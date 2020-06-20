# Keras-Model-Deployment

Often there’s a need to abstract away your machine learning model details and just deploy or integrate it with easy to use API endpoints. For eg., We can provide a URL endpoint using which anyone can make a POST request and they would get a JSON response of what the model has inferred without having to worry about its technicalities.

In this project, we will create a Flask server to deploy our MSIST classification artificial neural network (ANN) built in Keras. We will then create a simple Flask server which will accept POST request and do some image preprocessing and return the predictions.
