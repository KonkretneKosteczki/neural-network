import numpy as np
import json
from matplotlib.widgets import TextBox


class Visualization:
    def __init__(self, ax, color, means, samples, variance, modes):
        self.means = means
        self.samples = samples
        self.variance = variance
        self.modes = modes
        self.points = []
        self.data = self.generate_data(self.samples, self.means, self.variance)
        self.color = color
        self.ax = ax
        self.text_boxes = []
        self.plot = ax.scatter(*self.data, c=color)

    def draw(self):
        y, x = self.generate_data(self.samples, self.means, self.variance)
        self.plot.set_offsets(np.c_[x, y])
        # plt.subplot(111)
        # plt.draw()

    def display_inputs(self, plt, text_box_offset=0):
        self.text_boxes = [
            self.add_text_box(plt, [0.15 + text_box_offset, 0.150, 0.3, 0.04], 'Means', str(self.means),
                              self.update_means),
            self.add_text_box(plt, [0.15 + text_box_offset, 0.105, 0.3, 0.04], 'Samples', str(self.samples),
                              self.update_samples),
            self.add_text_box(plt, [0.15 + text_box_offset, 0.060, 0.3, 0.04], 'Variances', str(self.variance),
                              self.update_variance),
            self.add_text_box(plt, [0.15 + text_box_offset, 0.015, 0.3, 0.04], 'Modes', str(self.modes),
                              self.update_modes)
        ]

    def update_samples(self, text):
        samples = int(text)
        if samples != self.samples:
            self.samples = samples
            self.draw()

    def update_means(self, text):
        means = json.loads(text)
        if self.parse_range(means) and not np.array_equal(means, self.means):
            self.means = means
            self.draw()

    def update_variance(self, text):
        variance = json.loads(text)
        if self.parse_range(variance) and not np.array_equal(variance, self.variance):
            self.variance = variance
            self.draw()

    def update_modes(self, text):
        modes = int(text)
        if modes != self.modes:
            self.modes = modes
            self.draw()

    def add_text_box(self, plt, rect, label, initial_value, callback):
        text_box = TextBox(plt.axes(rect), label, initial=initial_value, color=self.color, hovercolor=self.color)
        text_box.on_submit(callback)
        return text_box

    def generate_data(self, samples, means, variance):
        x, y, self.points = [], [], []
        for i in range(self.modes):
            covariance = np.array([
                np.random.uniform(variance[0], variance[1], 2),
                np.random.uniform(variance[0], variance[1], 2)
            ])
            normal_distribution = np.random.multivariate_normal(
                np.array(np.random.uniform(means[0], means[1], 2)),
                np.dot(covariance, covariance.transpose()),
                samples)
            y.extend(normal_distribution[:, 0])
            x.extend(normal_distribution[:, 1])
            self.points.extend(normal_distribution)
        return y, x

    @staticmethod
    def parse_range(parsed_text):
        return isinstance(parsed_text, list) and all(isinstance(x, (int, float)) for x in parsed_text) and len(
            parsed_text) == 2
