import numpy as np


# Get euclidean distance between each data point
def euclidean_distance(vector1, vector2):
    distance = 0
    for i in range(len(vector1)):
        distance += np.square(float(vector1[i]) - float(vector2[i]))
    return np.sqrt(distance)


# Generate training dataset (100 points and 10 dimensions)
# Target column of first 50 points is 0, second is 1
def generate_training(dim):
    training = np.array(np.random.randint(2, size=(100, dim)))
    training[:50, 0] = 0
    training[50:, 0] = 1
    return training


# Generate test dataset
def generate_test(dim):
    return generate_training(dim)


# 1-NN Algorithm for each point
# Count all that are correct (check if training point has same value as test for dim0)
def knn(dim):
    distances = {}
    correct = 0
    training = generate_training(dim)
    test = generate_test(dim)

    for i in range(len(test)):
        for j in range(len(training)):
            distances[j] = euclidean_distance(training[j], test[i])

        sorted_dist = sorted(distances, key=distances.__getitem__)
        if training[sorted_dist[0]][0] == test[i][0]:
            correct += 1
    return correct


# Repeat 10 times and average results
def get_average(dims):
    count = 0
    for i in range(10):
        count += knn(dims)
    accuracy = float(count) / 1000
    print accuracy


def normalize_list(list_normal):
    max_value = max(list_normal)
    min_value = min(list_normal)
    for i in range(len(list_normal)):
        list_normal[i] = float(list_normal[i] - min_value) / (max_value - min_value)
    print(list_normal)
    return list_normal



def main():
    #get_average(10)
    # training = np.array([[1,.33, .15, 1, .17, .29],
    #                     [0,.8, .19, .9, .67, .71],
    #                     [0,.47, .35, 1, 0, 1],
    #                     [0,.53, .4, .95, .5, .43],
    #                     [1,.6, .1, .99, .83, .43],
    #                     [1,1, .15, 1, .5, .29],
    #                     ]
    #                     )
    #
    # validation = np.array([[0, 0, .21, 1, .33, .57],
    #                     [1, .73, .04, 1, 1, .29],
    #                     [1, .67, .05, 1, .83, 0],
    #                     [0, .2, .25, .94, .77, .86]])
    #
    # correct = knn(training, validation)
    # accuracy = float(correct) / len(validation)
    # print accuracy



    # input and output arrays
    x = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]).reshape((6, 2))
    y = np.array([-1, -1, -1, -2, -2, -1, 1, 1, 1, 2, 2, 1]).reshape((6, 2))

    # Calculate least squares
    W = np.linalg.lstsq(x, y)
    print(W)

if __name__ == "__main__":
    main()
