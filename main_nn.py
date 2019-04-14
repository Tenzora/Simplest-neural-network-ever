import numpy as np

class NN:
    def __init__(self):
        np.random.seed(1)
        self.weights = (2 * np.random.random((1, 3)) - 1)[0]

    def train(self, inputs, outputs, iteration):
        for _ in range(iteration):
            output = self.predict(inputs)
            error = outputs - output
            adj = 0.01 * np.dot(inputs.T, error)
            self.weights += adj

    def predict(self, inputs):
        res = []
        prod = np.dot(inputs, self.weights)
        for unrounded in prod:
            res.append(int(round(unrounded)))
        return res

#-----------------------------------------------------------------------------------#

train_pattern = np.array([
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 1],
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 0, 1]
])

train_labels = np.array([
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    0
]).T

predict_pattern = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [1, 1, 1],
    [0, 1, 0],
    [0, 0, 1]
])

nn = NN()
nn.train(train_pattern, train_labels, 30)
print(nn.predict(predict_pattern))

