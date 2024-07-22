import numpy as np

np.random.seed(1)
synapticWeights = 2 * np.random.random((3, 1)) - 1    # randomized the initial weights for the neural network

def step_function(x):   # activation function
    return 1 if x >= 0 else 0

# training dataset
trainingInputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
trainingOutputs = np.array([[0], [1], [1], [0]])

# training function
def train(inputs, outputs, weights, iterations):
    for _ in range(iterations):
        for i in range(len(inputs)):
            prediction = step_function(np.dot(inputs[i], weights))
            error = outputs[i] - prediction
            weights += error * inputs[i].reshape(3, 1)
    return weights

# train the neural network
synapticWeights = train(trainingInputs, trainingOutputs, synapticWeights, 10000)

# evaluate the new input with the model
new_input = np.array([1, 0, 0])
output = step_function(np.dot(new_input, synapticWeights))

print("Random starting synaptic weights: ")
print(2 * np.random.random((3, 1)) - 1)  # same as at the beginning
print("New synaptic weights after training: ")
print(synapticWeights)
print("Considering new situation [1, 0, 0] -> ?: ")
print(output)