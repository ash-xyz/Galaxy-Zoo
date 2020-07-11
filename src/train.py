import tensorflow as tf
from data import load_train_data
from model import create_model
import matplotlib.pyplot as plt


def train_model():
    """Trains and saves a keras model

    Returns:
        model: Trained Model
        history: history of losses
    """
    model = create_model()
    train_generator, valid_generator = load_train_data()

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer, loss='mse', metrics="accuracy")
    history = model.fit(
        x=train_generator, validation_data=valid_generator, epochs=1, steps_per_epoch=STEP_SIZE_TRAIN, validation_steps=STEP_SIZE_VALID)
    model.save("../models/model_1")
    # Get training and test loss histories
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')

    return model, history


model, history = train_model()
