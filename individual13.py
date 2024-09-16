import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load and preprocess the MNIST dataset
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data to [0, 1] and reshape to (28 * 28 = 784,)
    x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0

    return (x_train, y_train), (x_test, y_test)

# Build the model
def build_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(784,)))  # Input layer
    model.add(layers.Dense(30, activation='sigmoid'))  # Hidden layer (30 neurons)
    model.add(layers.Dense(10, activation='sigmoid'))  # Output layer (10 neurons)

    # Compile the model
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Main script
if __name__ == "__main__":
    # Load the data
    (x_train, y_train), (x_test, y_test) = load_data()

    # Build the model
    model = build_model()

    # Train the model
    model.fit(x_train, y_train, epochs=30, batch_size=10, validation_split=0.1)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test Accuracy: {test_acc}')
