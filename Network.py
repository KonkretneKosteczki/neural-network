import numpy as np
import Neuron


class Network(object):
    def __init__(self, layers, ax, x_lim, learning_rate=1, iterations=100):
        # layers is an array of length equivalent to number of layers,
        # where values of it's elements are the number of neurons in a layer
        self.layers = []
        self.iterations = iterations
        self.learning_rate = learning_rate
        for layer_index in range(len(layers)):
            self.layers.append([])
            number_of_neurons = layers[layer_index]
            inputs_number = 2
            if layer_index > 0:
                inputs_number = layers[layer_index-1]

            for neuron in range(number_of_neurons):
                self.layers[-1].append(
                    Neuron.Neuron(ax, x_lim, Neuron.Neuron.tanh, Neuron.Neuron.tanh_derivative,
                                  inputs_number, learning_rate))

    def solve(self, inputs):
        all_outputs = [inputs]
        last_layer_outputs = inputs

        for layer in self.layers:
            this_layer_outputs = []
            for neuron in layer:
                this_layer_outputs.append(neuron.solve(neuron.add_bias([last_layer_outputs]))[0])
            last_layer_outputs = this_layer_outputs
            all_outputs.append(this_layer_outputs)

        return all_outputs

    def train_once(self, train_inputs, train_outputs, iteration):
        learning_rate = self.learning_rate / (iteration + 1)  # variable learning rate
        all_network_outputs = self.solve(train_inputs)
        adjustments = []

        final_layer = self.layers[-1]
        final_layer_outputs = all_network_outputs[-1]

        adjustments.append([])
        previous_layer_errors = []
        for neuron_index in range(len(final_layer)):
            neuron = final_layer[neuron_index]
            error = train_outputs[neuron_index] - final_layer_outputs[neuron_index]
            a = neuron.activation_function_derivative([final_layer_outputs[neuron_index]])[0] * error
            previous_layer_errors.append(neuron.weights * a)
            layer_adjustment = (learning_rate * np.dot(neuron.add_bias([all_network_outputs[-2]]), a))[0]
            adjustments[-1].append(layer_adjustment)

        # back-propagation
        reverse_layers_no_final = self.layers[:-1][::-1]  # skip final layer, reverse order
        reverse_outputs_no_final = all_network_outputs[:-1][::-1]
        for reverse_layer_index in range(len(reverse_layers_no_final)):
            adjustments.append([])
            layer = reverse_layers_no_final[reverse_layer_index]
            layer_outputs = all_network_outputs[-2 - reverse_layer_index]
            layer_inputs = all_network_outputs[-3-reverse_layer_index]
            layer_errors = np.array(previous_layer_errors)
            previous_layer_errors = []
            for neuron_index in range(len(layer)):
                neuron = layer[neuron_index]
                error = sum(layer_errors[:, neuron_index])

                a = neuron.activation_function_derivative([layer_outputs[neuron_index]])[0] * error
                previous_layer_errors.append(neuron.weights * a)
                # possibly add bias dunno
                layer_adjustment = (learning_rate * np.dot(neuron.add_bias([layer_inputs]), a))[0]
                adjustments[-1].append(layer_adjustment)

        # apply adjustments
        for layer_adjustment_index in range(len(adjustments)):
            layer_adjustment = adjustments[layer_adjustment_index]
            layer = self.layers[-1 - layer_adjustment_index]

            for neuron_adjustment_index in range(len(layer_adjustment)):
                neuron_adjustment = layer_adjustment[neuron_adjustment_index]
                neuron = layer[neuron_adjustment_index]
                neuron.weights += neuron_adjustment

    def train_network(self, all_train_inputs, all_train_outputs):
        for iteration in range(self.iterations):
            for input_index in range(len(all_train_inputs)):
                self.train_once(all_train_inputs[input_index], all_train_outputs[input_index], iteration)
        print("trained")
