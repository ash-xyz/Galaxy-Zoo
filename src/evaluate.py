import tensorflow as tf
import numpy as np
import pandas as pd
from data import CLASSES, load_test_data, get_id


def generate_results(model_name):
    """Predicts the results using a certain model and outputs them to a predictions .csv

    Args:
        model_name: Name of a keras model, where the model is located in the models directory
    """
    test_gen = load_test_data()
    model = tf.keras.models.load_model("../models/" + model_name)

    predictions = model.predict(
        test_gen, steps=test_gen.n / test_gen.batch_size)
    header = open('../predictions/prediction_format.csv', 'r').readlines()[0]
    
    with open('../predictions/prediction_2.csv', 'w') as outfile:
        outfile.write(header)
        for i in range(len(test_gen.filenames)):
            id_ = (get_id(test_gen.filenames[i]))
            pred = predictions[i]
            outline = id_ + "," + ",".join([str(x) for x in pred])
            outfile.write(outline + "\n")


generate_results("best_model")
