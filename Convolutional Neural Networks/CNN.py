import pandas as pd
import numpy as np
from PIL import Image

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

# Compile the model and fit it to the data. 10 epochs is probably good. When I did this,
# I got a validation accuracy of around 67%.

# Now build a more complicated convolutional network. You might look at the example used on
# Fashion MNIST from the book for inspiration.

# Fit this more complicated model to the data. Plot the learning curve history. If you are able
# to get an accuracy over 80%, you’re good. If not, go back and tweak your model or hyperparameters.
# I expect some teams will be able to do significantly better than my result!

# Once you have your final model, train it one more time, being sure to save the learning curve
# history plot and to test it on the test data.
