import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from cost_function import CostFunction
from gradient_descent import GradientDescent
from gradient_descent_experiment import GradientDescentExperiment

from base_function import BaseFunction


class PolynomialRegressionExperiment(GradientDescentExperiment):
    def setup_func(self):
        self.real_theta = [2, 0.1, 0.1, 0.005]
        self.x = np.arange(-30, 20, 0.5)
        self.y = list(map(lambda u: u - 0*(np.random.rand()-0.5),
                          list(self.real_theta[0] +
                               self.real_theta[1]*self.x +
                               self.real_theta[2]*self.x**2 +
                               self.real_theta[3]*self.x**3))
                      )
        # our source data is not linear
        # so we add new features
        new_x = list(map(lambda val: [1, val, val**2, val**3], self.x))
        # but new features are to big
        # so we apply feature scaling
        means = [0, 0, 0, 0]
        mins = [new_x[0][0], new_x[0][1], new_x[0][2], new_x[0][3]]
        maxs = [new_x[0][0], new_x[0][1], new_x[0][2], new_x[0][3]]

        for j in range(0, len(new_x)):
            for i in range(1, len(mins)):
                e = new_x[j][i]
                means[i] = means[i] + e/len(new_x)
                if e < mins[i]:
                    mins[i] = e
                if e > maxs[i]:
                    maxs[i] = e
        for j in range(0, len(new_x)):
            for i in range(1, len(mins)):
                new_x[j][i] = (new_x[j][i] - means[i])/(maxs[i]-mins[i])
        # new_x = list(map(lambda val: [1, val[0], val[1]**2, val[2]**3], new_x))

        return CostFunction(new_x, self.y)

    def output_results(self):
        iterations = self.get_history_iterations()
        values = self.get_history_values()
        # c_v = np.frompyfunc(self.get_func(), 4, 1)
        # X = np.arange(-30, 20, 0.1)
        # Y = np.arange(-10, 10, 0.1)
        # X, Y = np.meshgrid(X, Y)
        # Z = c_v(X, Y)

        print('result theta')
        print(self.theta)
        print('real theta')
        print(self.real_theta)

        fig, ax = plt.subplots()
        ax.plot(self.x, self.y, 'b.')
        x1 = np.arange(-30, 20, 0.5)
        y1 = self.theta[0] + self.theta[1] * x1 + \
            self.theta[2] * x1**2+self.theta[3] * x1**3
        ax.plot(x1, y1, 'r.')
        plt.show()

        plt.style.use('_mpl-gallery')

        # x2 = np.arange(-10, 10, 0.25)
        # y2 = c_v(x2, 1)

        # fig, ax = plt.subplots()
        # ax.plot(x2, y2)
        # plt.show()

        # Plot the surface
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # ax.view_init(40, -10)
        # ax.plot_surface(X, Y, Z, cmap=cm.Blues)
        # ax.plot3D(list(map(lambda values: values[0], values)), list(map(
        #     lambda values: values[1], values)), list(map(lambda v: c_v(*v), values)), 'g.')

        # plt.show()

        fig, ax = plt.subplots()
        ax.plot(list(range(0, len(iterations))), iterations, 'b.')
        plt.show()


lre = PolynomialRegressionExperiment()
lre.set_start_theta([3, 1, 1, 1])
lre.set_max_iterations(10000)

rates = [0.000000001]

for rate in rates:
    lre.set_learning_rate(rate)
    lre.run_all()

# real_theta = [2, 0.1, 0.1, 0.005]
# x = np.arange(-30, 20, 0.5)
# y = list(map(lambda u: u - np.random.rand()-0.5,
#              list(real_theta[0] +
#                        real_theta[1]*x +
#                        real_theta[2]*x**2 +
#                        real_theta[3]*x**3))
#          )

# fig, ax = plt.subplots()
# ax.plot(x, y, 'b.')
# plt.show()
