class Network:
    def __init__(self, loss, loss_prime):
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # evaluate results for some data
    def evaluate(self, x_test, y_test):
        # sample dimension first
        samples = len(x_test)
        err = 0

        prediction = self.predict(x_test)

        # run network over all samples
        for i in range(samples):

            example_error = self.loss(y_test[i], prediction[i])

            err = err + example_error

        return err / samples

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # Ensure loss function is set
        if self.loss is None:
            raise ValueError(
                "Loss function is not set. Use the 'use' method to set the loss and loss_prime."
            )
        if self.loss_prime is None:
            raise ValueError(
                "Loss function is not set. Use the 'use' method to set the loss and loss_prime."
            )
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err = err + self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples

            if (i + 1) % round(epochs / 10) == 0 or i == 0:
                print("For the epoch %d/%d   the error is %f" % (i + 1, epochs, err))
