from activation_Layer import ActivationLayer
from attention import AttentionLayer
from layers import FCLayer, SoftmaxLayer, relu, relu_prime
from losses import cross_entropy, cross_entropy_prime
from mnist_loader import finish_up_data
from network import Network

# Create the network
net = Network(cross_entropy, cross_entropy_prime)
net.add(FCLayer(28 * 28, 100))  # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(relu, relu_prime))
net.add(AttentionLayer(100, 50))  # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(relu, relu_prime))
net.add(FCLayer(50, 10))  # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(SoftmaxLayer())


x_train, y_train, x_test, y_test = finish_up_data()
examples = 4000
epochs = 12
learning_rate = 0.4

# train the network
net.fit(
    x_train[:examples], y_train[:examples], epochs=epochs, learning_rate=learning_rate
)

# evaluate on test data
test_loss = net.evaluate(x_test[:100], y_test[:100])
print("Test loss:", test_loss)
