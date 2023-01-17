import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier

import pickle

def predict(arr):
    # Load the model
    with open('final_model.sav', 'rb') as f:
        model = pickle.load(f)
    classes = {0:'Diabetes',1:'No Diabetes'}

    # return prediction as well as class probabilities

    preds = model.predict_proba([arr])[0]

    return (classes[np.argmax(preds)], preds)


