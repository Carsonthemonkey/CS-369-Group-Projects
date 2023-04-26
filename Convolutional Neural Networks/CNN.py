import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path

IMAGES_PATH = Path() / "graphs"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# Read /home/labs/drake/cs369/answers.csv into a pandas dataframe
answers = pd.read_csv('/home/labs/drake/cs369/answers.csv')

# Remove the two entries that don’t have images.
nonexistent = ['AB-0657+10345', 'AB-1011+10371']
answers = answers[~answers['BallotID'].isin(nonexistent)]

# Read all of the images (which are in /home/labs/drake/cs369/writein_crops) into a single, three-dimensional numpy array.
m = len(answers)
x = np.empty((m, 53, 358), dtype=float)
i = 0
for id in answers['BallotID']:
    img = np.array(Image.open(
        '/home/labs/drake/cs369/writein_crops/' + id + '.jpg'))
    x[i, :, :] = img[:53, :358] / 255.0
    i += 1

# Set aside 20% of the data for testing.
p = int(m * .2)
train = answers[:-p]

X_train_full, y_train_full = x[:-p], answers['raiford'][:-p]
X_test, y_test = x[-p:], answers['raiford'][-p:]

# Set aside 20% of the remaining data for validation. At this point, your training X values
# should have shape (30528, 53, 358) and your training y value should have shape (30528,).
p = int(len(X_train_full) * .2)
X_train, y_train = X_train_full[:-p], y_train_full[:-p]
X_valid, y_valid = X_train_full[-p:], y_train_full[-p:]
print('SHAPES   train x:  ', np.shape(X_train),
      '        train y: ', np.shape(y_train))

# Build a very simple network that has only a single output neuron with sigmoid activation.
tf.random.set_seed(42)
np.random.seed(42)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the model and fit it to the data. 10 epochs is probably good. When I did this,
# I got a validation accuracy of around 67%.
model.compile(loss="binary_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid), verbose=2)

# Now build a more complicated convolutional network. You might look at the example used on
# Fashion MNIST from the book for inspiration.

# see book pg 496
DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3,
                        padding="same", activation="relu", kernel_initializer="he_normal")
model = tf.keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[53, 358, 1]),
    tf.keras.layers.MaxPooling2D(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPooling2D(),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=1, activation="sigmoid")  # job 528, 531
])

# Fit this more complicated model to the data. Plot the learning curve history. If you are able
# to get an accuracy over 80%, you’re good. If not, go back and tweak your model or hyperparameters.
# I expect some teams will be able to do significantly better than my result!
model.compile(loss="binary_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid), verbose=2)

# Once you have your final model, train it one more time, being sure to save the learning curve
# history plot and to test it on the test data.
pd.DataFrame(history.history).plot(
    figsize=(8, 5), xlim=[0, 11], ylim=[0, 1], grid=True, xlabel="Epoch",
    style=["r--", "r--.", "b-", "b-*"])
plt.legend(loc="lower left")  # extra code
save_fig("keras_learning_curves_plot")  # extra code
plt.show()

model.evaluate(X_test, y_test)
