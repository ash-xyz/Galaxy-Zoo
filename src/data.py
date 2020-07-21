import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot
CLASSES = ['Class1.1', 'Class1.2', 'Class1.3', 'Class2.1', 'Class2.2', 'Class3.1', 'Class3.2', 'Class4.1', 'Class4.2', 'Class5.1', 'Class5.2', 'Class5.3', 'Class5.4', 'Class6.1', 'Class6.2', 'Class7.1', 'Class7.2', 'Class7.3', 'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7', 'Class9.1', 'Class9.2', 'Class9.3', 'Class10.1', 'Class10.2', 'Class10.3', 'Class11.1', 'Class11.2', 'Class11.3', 'Class11.4', 'Class11.5', 'Class11.6'
           ]
TRAIN_LABEL_DIR = '../data/training_solutions_rev1.csv'
TRAIN_IMAGES_DIR = '../data/images_training_rev1/'
TEST_LABEL_DIR = '../predictions/prediction_format.csv'
TEST_IMAGES_DIR = '../data/images_test_rev1/'
IMAGE_SHAPE = (224, 224, 3)
BATCH_SIZE = 32


def append_ext(id):
    """Appends jpg to a number"""
    return str(id)+".jpg"


def get_id(fname):
    return fname.replace(".jpg", "")


def load_train_data():
    """ Loads the training data and it's labels

    Returns:
        train_generator: training data
        validation_generator: validation data
    """
    # Dataframe
    train_data = pd.read_csv(TRAIN_LABEL_DIR)
    train_data["GalaxyID"] = train_data["GalaxyID"].apply(append_ext)
    # Generator
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255., validation_split=0.05,rotation_range=90,horizontal_flip=True,vertical_flip=True)
    # Training Generator
    train_generator = generator.flow_from_dataframe(dataframe=train_data, directory=TRAIN_IMAGES_DIR, x_col="GalaxyID", y_col=CLASSES, subset="training", batch_size=BATCH_SIZE, seed=42,
                                                    shuffle=True,
                                                    class_mode="raw",
                                                    target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    # Validation Generator
    validation_generator = generator.flow_from_dataframe(dataframe=train_data, directory=TRAIN_IMAGES_DIR, x_col="GalaxyID", y_col=CLASSES, subset="validation", batch_size=BATCH_SIZE, seed=42,
                                                         shuffle=True,
                                                         class_mode="raw",
                                                         target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    return train_generator, validation_generator


def load_test_data():
    """Loads the test data

    Returns:
        test_generator: generator for the test images
    """
    test_data = pd.read_csv(TEST_LABEL_DIR)
    test_data["GalaxyID"] = test_data["GalaxyID"].apply(append_ext)

    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255.)
    test_generator = generator.flow_from_dataframe(
        dataframe=test_data,
        directory=TEST_IMAGES_DIR,
        x_col="GalaxyID",
        y_col=None,
        batch_size=BATCH_SIZE,
        seed=42,
        shuffle=False,
        class_mode=None,
        target_size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
    return test_generator
