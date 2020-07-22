import tensorflow as tf
from data import CLASSES, IMAGE_SHAPE


def create_model():
    """Creates a keras model for training

    Returns:
        model: keras model
    """
    # Densenet Model
    conv_net = tf.keras.applications.MobileNetV2(
        include_top=False, input_shape=IMAGE_SHAPE)
    neural_net = tf.keras.layers.Flatten()(conv_net.output)
    neural_net = tf.keras.layers.Dropout(rate=0.5)(neural_net)
    neural_net = tf.keras.layers.Dense(300, activation='relu')(neural_net)
    neural_net = tf.keras.layers.Dropout(rate=0.5)(neural_net)
    neural_net = tf.keras.layers.Dense(
        len(CLASSES), activation='sigmoid')(neural_net)
    model = tf.keras.Model(inputs=conv_net.inputs, outputs=neural_net)
    for layer in model.layers:
        layer.trainable = True
    return model
