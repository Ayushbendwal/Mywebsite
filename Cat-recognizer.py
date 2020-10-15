import numpy as np
import random
import h5py
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from scipy import ndimage

# Data Preprocessing START HERE


# Splitting Datasets into train/test
train_dataset = h5py.File("DataSets/train_catvnoncat.h5", "r")
x_train = np.array(train_dataset["train_set_x"][:])
y_train = np.array(train_dataset["train_set_y"][:])

test_datasets = h5py.File("DataSets/test_catvnoncat.h5", "r")
x_test = np.array(test_datasets["test_set_x"][:])
y_test = np.array(test_datasets["test_set_y"][:])

classes = np.array(test_datasets["list_classes"][:])

y_train = y_train.reshape((1, y_train.shape[0]))
y_test = y_test.reshape((1, y_test.shape[0]))

# example of a picture
# index = random.randint(1, x_train.shape[1])
# plt.imshow(x_train[index])
# plt.xlabel("it's a '" +
#            classes[np.squeeze(y_train[:, index])].decode("utf-8"))
# plt.show()


# finding train, test data number
m_train = x_train.shape[0]
m_test = x_test.shape[0]
num_px = x_train.shape[1]

# Resahping the Train and Test Data
x_train_flatten = x_train.reshape(m_train, num_px*num_px*3).T
x_test_flatten = x_test.reshape(m_test, num_px*num_px*3).T

# print("Flatten shape x_train:", x_train_flatten.shape)
# print("Flatten shape X_test :", x_test_flatten.shape)

x_train = x_train_flatten/255
x_test = x_test_flatten/255

# DATA Preprocessing ends Here

# Helper Functions


def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s


def intialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]

    # Forward Propogation TO find Cost

    A = sigmoid(np.dot(w.T, X) + b)  # Relu activation
    cost = np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))/-m  # Cost
    cost = np.squeeze(cost)  # To remove unwanted Entries

    # Backward Propogation TO find Gradient
    dw = np.dot(X, (A-Y).T)/m
    db = np.sum(A-Y)/m

    gradients = {"dw": dw, "db": db}

    return gradients, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):
        gradients, cost = propagate(w, b, X, Y)

        dw = gradients["dw"]
        db = gradients["db"]

        # updating parameter
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # to update cost after 100 iterations
        if i % 100 == 0:
            costs.append(cost)

        # to print cost after 100 iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i : %f" % (i, cost))

    params = {"w": w, "b": b}
    gradients = {"dw": dw, "db": db}

    return params, gradients, costs


def predict(w, b, X):

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction


def model(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    w, b = intialize_with_zeros(x_train.shape[0])
    paramters, gradients, costs = optimize(
        w, b, x_train, y_train, num_iterations, learning_rate, print_cost)

    w = paramters["w"]
    b = paramters["b"]

    Y_prediction_train = predict(w, b, x_train)
    Y_prediction_test = predict(w, b, x_test)

    data = {"w": w,
            "b": b,
            "costs": costs,
            "learning_rate": learning_rate,
            "num_iterations": num_iterations,
            "Y_prediction_train": Y_prediction_train,
            "Y_prediction_test": Y_prediction_test}

    print("train accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_test - y_test)) * 100))

    return data


if __name__ == "__main__":
    data = model(x_train, y_train, x_test, y_test, 2000, 0.005, True)

    test_image = "test.jpg"
    image = np.array(mpimg.imread(test_image))
    image = image/255
    my_image = resize(image, (64, 64)).reshape((1, 64*64*3)).T
    my_predicted_image = predict(data["w"], data["b"], my_image)

    plt.imshow(image)
    plt.xlabel("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" +
               classes[int(np.squeeze(my_predicted_image)), ].decode("utf-8") + "\" picture.")
    plt.show()
