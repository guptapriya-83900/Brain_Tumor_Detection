import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","training", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
         # Convert to grayscale if the model expects grayscale images
        test_image = test_image.convert('L')  # 'L' mode is for grayscale conversion

        # Convert image to array and add batch dimension
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Predict
        result = np.argmax(model.predict(test_image), axis=1)

        print(result)

        if result[0] == 1:
            prediction = 'No Tumor'
            return [{ "image" : prediction}]
        else:
            prediction = 'Brain Tumor'
            return [{ "image" : prediction}]
