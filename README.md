# Website to recognize your Facial Expression
### Website link - https://practice-facial-expressions.herokuapp.com/
### This is a demo website deployed on heroku using python/flask to use the Facial Expression Recognition model.
The Jupyter notebook for the model is available in this repo.

The dataset used to train the model is FER2013 and is trained on a CNN which gives a training accuracy of 74% and validation accuracy of 63%.
The client side uses JQuery to send the Image captured to the server where the we use
opencv haar cascade face detection to detect the face and crop it then process it and pass it through the CNN.
Then the predicted emotion is written on the Image and sent back to the client side where its displayed.

## Some limitations of the model are:
- Profile view of faces cannot be detected as we are using haarcascade_frontalface_default.xml for face detection.
- Improper illumination of face like half illuminated, insifficient light, glare or shadows can lead to inaccuracy. 
- Low camera quality and resolution may also cause some inaccuracy.
