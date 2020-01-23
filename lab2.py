import Network
import PointClass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# defaults
defModes = 1
defMeans = [-8, 8]
defVariance = [-1, 1]
defSamples = 10
x_lim = (-10, 10)
y_lim = (-10, 10)

# init
fig, ax = plt.subplots()
ax.set(xlim=x_lim, ylim=y_lim)
plt.subplots_adjust(bottom=0.25)
mesh = [
    ax.plot([x_lim[0]], [x_lim[1]], color="lightcoral", marker=".", linewidth=0, alpha=0.3)[0],
    ax.plot([x_lim[0]], [x_lim[1]], color="lightskyblue", marker=".", linewidth=0, alpha=0.3)[0]
]


def clear_mesh():
    x = [x_lim[0]]
    y = [y_lim[0]]
    mesh[0].set_ydata(y)
    mesh[0].set_xdata(x)
    mesh[1].set_ydata(y)
    mesh[1].set_xdata(x)


class1 = PointClass.Visualization(ax, "lightcoral", defMeans, defSamples, defVariance, defModes)
class1.display_inputs(plt, 0)

class2 = PointClass.Visualization(ax, "lightskyblue", defMeans, defSamples, defVariance, defModes)
class2.display_inputs(plt, 0.45)


def draw_mesh():
    xs = np.linspace(*x_lim)
    ys = np.linspace(*y_lim)
    class1_x = [];
    class1_y = [];
    class2_x = [];
    class2_y = []
    mesh_coordinates = [[x, y] for x in xs for y in ys]
    network_solutions = [network.solve(xy) for xy in mesh_coordinates]
    # print(solutions)
    for solution_index in range(len(network_solutions)):
        solution = network_solutions[solution_index]
        x, y = mesh_coordinates[solution_index]

        if solution[0] > solution[1]:
            class2_x.append(x)
            class2_y.append(y)
        elif solution[0] < solution[1]:
            class1_x.append(x)
            class1_y.append(y)

    mesh[0].set_ydata(class1_y)
    mesh[0].set_xdata(class1_x)
    mesh[1].set_ydata(class2_y)
    mesh[1].set_xdata(class2_x)


def true_output():
    all_modes = class1.modes + class2.modes
    output = []
    for index in range(class1.modes):
        out = np.zeros(all_modes)
        out[index] = 1
        for i in range(class1.samples):
            output.append(out)

    for index in range(class2.modes):
        out = np.zeros(all_modes)
        out[class1.modes + index] = 1
        for i in range(class2.samples):
            output.append(out)

    return output


network = Network.Network([2, 3, 2], ax, x_lim, 1)


# neuron = Neuron.Neuron(ax, x_lim, Neuron.Neuron.heaviside_step, 1)


def train(_):
    clear_mesh()
    # neuron.weights = np.random.uniform(0, 1, 3)
    # neuron.train(
    #     np.concatenate((class1.points, class2.points), axis=0),
    #     true_output(), neuron.heaviside_step, neuron.heaviside_step_derivative,
    #     iterations=400
    # )
    network.train_network(
        np.concatenate((class1.points, class2.points), axis=0),
        true_output()
    )
    draw_mesh()
    plt.draw()


def draw(_):
    class1.draw()
    class2.draw()
    # neuron.clear()
    clear_mesh()
    plt.draw()


# noinspection PyTypeChecker
buttonT = Button(plt.axes((0, 1 - 0.075, 0.1, 0.075)), "train")
buttonT.on_clicked(train)

# noinspection PyTypeChecker
buttonD = Button(plt.axes((0.1, 1 - 0.075, 0.1, 0.075)), "draw")
buttonD.on_clicked(draw)

plt.show()
