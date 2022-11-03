# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:50:43 2022

@author: poojitha
"""

import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model,load_weights

# In[2]:
from keras.preprocessing.image import load_img, img_to_array

# In[3]:

img_width, img_height = 256, 256

# In[4]:
def preprocess_image(path):
    img = load_img(path, target_size = (img_height, img_width))
    a = img_to_array(img)
    a = np.expand_dims(a, axis = 0)
    a /= 255.
    return a





# In[5]:
test_images_dir = 'C:/Users/poojitha/Desktop/Classification/testset/'

model_path = r'C:\Users\poojitha\Desktop\Classification\Pred\CNN_model_256.h5'
#model_weights = r'C:\Users\poojitha\Desktop\Classification\Pred\CNN_weights_256.h5'

model = load_model(model_path)

#classes = {'junk': 0, 'temp': 1, 'tiltted': 2}
classes = ['junk','temp','tiltted']

with open('results.txt','a+')as f:
    for img in os.listdir(test_images_dir):
        img_path = os.path.join(test_images_dir,img)
        imgarr = preprocess_image(img_path)
        prediction = model.predict(imgarr)
        class_index = np.argmax(prediction)
        class_name = classes[class_index]
        conf_score = prediction[0][class_index]
        #print(img, class_name,conf_score)
        f.write(img + "," + class_name+ ","+ str(conf_score)+ "\n")
        
    