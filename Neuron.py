import numpy as np


class Neuron:
    def __init__(self, ax, x_lim, learning_rate=1):
        self.learningRate = learning_rate
        self.ax = ax
        self.x_lim = x_lim

        self.weights = np.random.uniform(-1, 1, 3)
        self.plot, = ax.plot([], [])
        self.border = []

    def get_boundary(self):
        b = self.weights[0]
        w2 = self.weights[1]
        w1 = self.weights[2]

        x = np.linspace(self.x_lim[0], self.x_lim[1], 2)
        y = (-(b / w2) / (b / w1)) * x + (-b / w2)

        return x, y

    def clear(self):
        for border in self.border:
            border.remove()
        self.border = []
        self.plot.set_ydata([])
        self.plot.set_xdata([])

    def draw(self):
        self.clear()
        x, y = self.get_boundary()
        self.plot.set_ydata(y)
        self.plot.set_xdata(x)
        self.border = [
            self.ax.fill_between(x, y, self.x_lim[0], alpha=0.3,
                                 facecolor="lightcoral" if self.weights[1] > 0 else "lightskyblue"),
            self.ax.fill_between(x, y, self.x_lim[1], alpha=0.3,
                                 facecolor="lightskyblue" if self.weights[1] > 0 else "lightcoral")
        ]

    def train(self, train_inputs, train_outputs, activation_function, activation_function_derivative, iterations=100):
        inputs_with_bias = np.array([[-1, *train_input] for train_input in train_inputs])

        for iteration in range(iterations):
            output = activation_function(np.dot(inputs_with_bias, self.weights))
            error = train_outputs - output
            adjustment = self.learningRate / (iteration + 1) * np.dot(inputs_with_bias.T,
                                                                      error * activation_function_derivative(output))
            self.weights += adjustment

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        return 1.0 - np.tanh(x) ** 2

    @staticmethod
    def heaviside_step(s):
        def single_heavy(value):
            return 1 if value > 0 else 0

        return list(map(single_heavy, s))

    @staticmethod
    def heaviside_step_derivative(s):
        return list(np.ones(len(s)))

    @staticmethod
    def sigmoid(s):
        return 1 / (1 + np.exp(-s))

    @staticmethod
    def sigmoid_derivative(s):
        sigmoid = Neuron.sigmoid(s)
        return sigmoid * (1 - sigmoid)

    @staticmethod
    def sin(s):
        return np.sin(s)

    @staticmethod
    def sin_derivative(s):
        return np.cos(s)

    @staticmethod
    def sign(s):
        def single_sign(value):
            if value < 0:
                return -1
            if value == 0:
                return 0
            return 1

        return list(map(single_sign, s))

    @staticmethod
    def sign_derivative(s):
        return Neuron.heaviside_step_derivative(s)

    @staticmethod
    def relu(s):
        def single_relu(value):
            return 0 if value <= 0 else value

        return list(map(single_relu, s))

    @staticmethod
    def relu_derivative(s):
        def single_relu_derivative(value):
            return 0 if value <= 0 else 1

        return list(map(single_relu_derivative, s))

    @staticmethod
    def leaky_relu(s):
        def single_leaky_relu(value):
            return 0.01 * value if value <= 0 else value

        return list(map(single_leaky_relu, s))

    @staticmethod
    def leaky_relu_derivative(s):
        def single_leaky_relu_derivative(value):
            return 0.01 if value <= 0 else 1

        return list(map(single_leaky_relu_derivative, s))
