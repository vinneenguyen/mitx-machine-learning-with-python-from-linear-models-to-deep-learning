from string import punctuation, digits

import numpy as np
import matplotlib.pyplot as plt
import random

### Part I

def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
      feature_vector - A numpy array describing the given data point.
      label - A real valued number, the correct classification of the data
      point.
      theta - A numpy array describing the linear classifier.
      theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    # Your code here
    pred = np.dot(theta, feature_vector) + theta_0
    return max(0, 1-label * pred)


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
      feature_matrix - A numpy matrix describing the given data. Each row
          represents a single data point.
      labels - A numpy array where the kth element of the array is the
          correct classification of the kth row of the feature matrix.
      theta - A numpy array describing the linear classifier.
      theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    # Your code here
    loss = []
    for i in range(feature_matrix.shape[0]):
        loss.append(hinge_loss_single(feature_matrix[i], labels[i], theta, theta_0))
    return np.mean(loss)


def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    pred = np.dot(current_theta, feature_vector) + current_theta_0
    if pred * label <= 0:
        current_theta += label * feature_vector
        current_theta_0 += label
    return current_theta, current_theta_0

def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    # Your code here
    theta = np.zeros(feature_matrix.shape[1], dtype=np.float64)
    theta_0 = 0
    for t in range(T):
        for i in range(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
    return theta, theta_0


def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    # Your code here
    theta = np.zeros(feature_matrix.shape[1], dtype=np.float64)
    theta_0 = 0.
    
    avg_theta = np.zeros(feature_matrix.shape[1], dtype=np.float64)
    avg_theta_0 = 0.
    for t in range(T):
        for i in range(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            avg_theta += theta
            avg_theta_0 += theta_0
    return avg_theta/(T*feature_matrix.shape[0]), avg_theta_0/(T*feature_matrix.shape[0])


def my_pegasos_single_step_update(feature_vector, label, L, eta, current_theta, current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    pred = np.dot(feature_vector, current_theta) + current_theta_0

    if pred * label <= 1:
        current_theta = (1-eta*L) * current_theta + eta*label*feature_vector
        current_theta_0 += eta*label
    else:
        current_theta = (1-eta*L) * current_theta
    return current_theta, current_theta_0

def pegasos_single_step_update(feature_vector, label, L, eta, current_theta, current_theta_0):
    """
    Section 1.5
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    mult = 1 - (eta * L)
    if label*(np.dot(feature_vector, current_theta) + current_theta_0) <= 1:
        return ((mult*current_theta) + (eta * label * feature_vector),
            (current_theta_0) + (eta*label))
    return (mult*current_theta, current_theta_0)


def pegasos(feature_matrix, labels, T, L):
    # Your code here
    random.seed(0)

    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0.
    cnt = 1.
    for i in range(T):
        indices = [k for k in range(len(feature_matrix))]
        random.shuffle(indices)
        for j in range(len(feature_matrix)):
            # Your code here
            chosen = indices[j]
            eta = 1./np.sqrt(cnt)
            #print(eta)
            cnt+=1
            theta, theta_0 = pegasos_single_step_update(feature_matrix[chosen], labels[chosen], L, eta, theta, theta_0)
    return theta, theta_0

def pegasos_sol(feature_matrix, labels, T, L):
    random.seed(0)

    current_theta = np.zeros(len(feature_matrix[0]))
    current_theta_offset = 0
    for i in range(T):
        indices = [k for k in range(len(feature_matrix))]
        random.shuffle(indices)
        for j in range(len(feature_matrix)):
            chosen = indices[j]
            eta = 1.0 / np.sqrt(i*len(feature_matrix) + j+1)
            #print()
            (current_theta, current_theta_offset) = pegasos_single_step_update(
                feature_matrix[chosen], labels[chosen], L, eta, current_theta,
                current_theta_offset)
    return (current_theta, current_theta_offset)


### Part II

def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
      feature_matrix - A numpy matrix describing the given data. Each row
          represents a single data point.
              theta - A numpy array describing the linear classifier.
      theta - A numpy array describing the linear classifier.
      theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is the predicted
    classification of the kth row of the feature matrix using the given theta
    and theta_0. If a prediction is GREATER THAN zero, it should be considered a positive
    classification.
    """
    # Your code here
    result = np.zeros(feature_matrix.shape[0])
    for i in range(feature_matrix.shape[0]):
        pred = np.dot(theta, feature_matrix[i]) + theta_0
        if pred > 0:
            result[i] = 1
        else:
            result[i] = -1
    return result


def perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    """
    Trains a linear classifier using the perceptron algorithm with a given T
    value. The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
      perceptron - An implementation of the perceptron algorithm to be used in
          this function. Note: this argument is not present in the local
          implementations of the accuracy functions, but is necessary for the
          online grader to work here. Locally you can use the perceptron
          function you define in your project1.py file.
      train_feature_matrix - A numpy matrix describing the training
          data. Each row represents a single data point.
      val_feature_matrix - A numpy matrix describing the training
          data. Each row represents a single data point.
      train_labels - A numpy array where the kth element of the array
          is the correct classification of the kth row of the training
          feature matrix.
      val_labels - A numpy array where the kth element of the array
          is the correct classification of the kth row of the validation
          feature matrix.
      T - The value of T to use for training with the perceptron algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    # Your code here
    theta, theta_0 = perceptron(train_feature_matrix, train_labels, T)

    train_predictions = classify(train_feature_matrix, theta, theta_0)
    val_predictions = classify(val_feature_matrix, theta, theta_0)

    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(val_predictions, val_labels)

    return (train_accuracy, validation_accuracy)

def average_perceptron_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T):
    """
    Trains a linear classifier using the average perceptron algorithm with
    a given T value. The classifier is trained on the train data. The
    classifier's accuracy on the train and validation data is then returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the average perceptron
            algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    # Your code here
    theta, theta_0 = average_perceptron(train_feature_matrix, train_labels, T)

    train_predictions = classify(train_feature_matrix, theta, theta_0)
    val_predictions = classify(val_feature_matrix, theta, theta_0)

    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(val_predictions, val_labels)

    return (train_accuracy, validation_accuracy)

def pegasos_accuracy(train_feature_matrix, val_feature_matrix, train_labels, val_labels, T, L):
    """
    Trains a linear classifier using the pegasos algorithm with 
    given T and L values. The classifier is trained on the train data.
    The classifier's accuracy on the train and validation data is then
    returned.

    Args:
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        T - The value of T to use for training with the algorithm.
        L - The value of L to use for training with the Pegasos algorithm.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the accuracy
    of the trained classifier on the validation data.
    """
    # Your code here
    theta, theta_0 = pegasos(train_feature_matrix, train_labels, T, L)

    train_predictions = classify(train_feature_matrix, theta, theta_0)
    val_predictions = classify(val_feature_matrix, theta, theta_0)

    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(val_predictions, val_labels)

    return (train_accuracy, validation_accuracy)

def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()

def bag_of_words(texts, remove_stopword=False):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Section 3 (e.g. remove stopwords, add bigrams etc.)
    """
    stopword = set()
    if remove_stopword:
        with open('stopwords.txt') as fp:
            for line in fp:
                word = line.strip()
                stopword.add(word)

    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary:
                if word in stopword:
                    continue
                dictionary[word] = len(dictionary)

    return dictionary

def extract_bow_feature_vectors(reviews, dictionary, binarize=True):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.
    """

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)], dtype=np.float64)

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                if binarize:
                    feature_matrix[i, dictionary[word]] = 1
                else:
                    feature_matrix[i, dictionary[word]] += 1
    return feature_matrix


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
