#WHEEEEEEEEEEEEEEEEEEEEEEEE
# :)
# Read /home/labs/drake/cs369/answers.csv into a pandas dataframe (should have a length of 47703)

# Remove the two entries that don’t have images.

# Read all of the images (which are in /home/labs/drake/cs369/writein_crops) into a single, three-dimensional numpy array.

# Set aside 20% of the data for testing.

# Set aside 20% of the remaining data for validation. At this point, your training X values 
# should have shape (30528, 53, 358) and your training y value should have shape (30528,).

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
