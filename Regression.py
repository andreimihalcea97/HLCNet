import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LinearRegression:
    def __init__(self, eta=0.001, epochs=10000):
        self.w = None  # The regression coefficient (slope of the best fit)
        self.b = None  # Y-axis intercept
        self.eta = eta  # The learning rate
        self.epochs = epochs  # The number of training epochs
        self.loss = None  # Loss function value

    def load_dataset(self):
        pass

    def predict(self, x):
        y = np.dot(x, self.w) + self.b
        return y

    @staticmethod
    def mean_sq_error(y_target, y_pred):
        residual = y_pred - y_target
        mse = np.mean(residual ** 2)
        return mse

    def optimize(self, x, y_target, y_pred):
        residual = y_target - y_pred
        w_derivate = -1 * np.dot(x.T, residual) / x.shape[0]
        b_derivate = -1 * np.mean(residual)
        self.w = self.w - self.eta * w_derivate
        self.b = self.b - self.eta * b_derivate

    def fit(self, x, y, w_from_zero=False):
        n_features = x.shape[1]
        self.w = np.zeros(n_features) if w_from_zero else np.random.randn(n_features)
        self.b = 0.0 if w_from_zero else np.random.randn()
        for ep in range(self.epochs):
            y_pred = self.predict(x)
            if len(y_pred.shape) == 1:  # We need 2D arrays for consistency across simple and multiple regressions
                y_pred = y_pred[:, np.newaxis]  # TO FIND A BETTER WAY TO MATCH THE SHAPES
            loss = self.mean_sq_error(y, y_pred)
            print(f"Loss after epoch {ep} = {loss}")
            self.optimize(x, y, y_pred)


def read_csv(csv_path, fields):
    return pd.read_csv(csv_path, usecols=fields)


if __name__ == '__main__':
    # Generate test dataset
    # np.random.seed(42)  # For reproducibility
    # x = np.linspace(1, 10, 500)  # Generate 100 evenly spaced values between 1 and 100
    # x = x.reshape(x.shape[0], 1)
    # print(x.shape)
    # true_slope = 2
    # true_intercept = 3
    # noise = np.random.normal(0, 1.5, size=x.shape)  # Add some noise
    # y = true_slope * x + true_intercept + noise  # Linear relationship with noise
    # y = y.reshape(y.shape[0], 1)
    # x_norm = 1/x
    # y = 1/y

    df = read_csv("datasets/regression/Salary_dataset.csv", ["YearsExperience", "Salary"])
    x = df["YearsExperience"].array[:, np.newaxis]
    y = df["Salary"].array[:, np.newaxis]
    # x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    print(x.shape)
    print(y.shape)
    # quit()

    # Plot the test dataset
    plt.scatter(x, y, color="blue", label="Data Points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Dataset")
    plt.legend()
    plt.show()

    # Train a linear regression model
    regression = LinearRegression()
    regression.fit(x, y, w_from_zero=False)
    pred = regression.predict(x)
    regression_line = np.full(x.shape, pred)

    # Plot the fitted regression line
    plt.scatter(x, y, color="blue", label="Data Points")
    plt.plot(x, regression_line, color="red", label="Regression Line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.show()
