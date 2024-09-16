import numpy as np
import pickle
import gzip
from network import Network, sigmoid, sigmoid_prime  # Ensure this import is correct

def load_data():
    """Load the MNIST data."""
    f = gzip.open('C:/Users/DELL/Downloads/neural-networks-and-deep-learning-master/data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()

    # Vectorize the labels (y)
    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorize(y) for y in training_data[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_results = [vectorize(y) for y in validation_data[1]]
    validation_data = list(zip(validation_inputs, validation_results))

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_results = [vectorize(y) for y in test_data[1]]
    test_data = list(zip(test_inputs, test_results))

    return training_data, validation_data, test_data
    
def load_data_wrapper():
    """Return a tuple containing `(training_data, validation_data, test_data)`."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [x for x, y in tr_d]
    training_results = [y for x, y in tr_d]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [x for x, y in va_d]
    validation_results = [y for x, y in va_d]
    validation_data = list(zip(validation_inputs, validation_results))

    test_inputs = [x for x, y in te_d]
    test_results = [y for x, y in te_d]
    test_data = list(zip(test_inputs, test_results))
    return (training_data, validation_data, test_data)
def vectorize(y):
    """Convert a digit into a 10-dimensional vector with a 1 in the position of the digit."""
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e

def main():
    training_data, validation_data, test_data = load_data_wrapper()
    net = Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

if __name__ == "__main__":
    main()
