import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=1):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loop(X)
        else:
            distances = self.compute_distances_two_loops(X)
        

        return self.predict_labels(distances)


    def compute_distances_two_loops(self, X_test):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        distance_matrix = np.zeros((X_test.shape[0], self.train_X.shape[0]))
        for i in range(X_test.shape[0]):
            for j in range(self.train_X.shape[0]):
                distance_matrix[i, j] = np.sum(np.abs(X_test[i] - self.train_X[j]))
        return distance_matrix
        

    def compute_distances_one_loop(self, X_test):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        distance_matrix = np.zeros((X_test.shape[0], self.train_X.shape[0]))
        for i in range(X_test.shape[0]):
            distance_matrix[i, :] = np.sum(np.abs(X_test[i] - self.train_X), axis=1)
        return distance_matrix
        

    def compute_distances_no_loops(self, X_test):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        
        return np.abs(X_test[:, None] - self.train_X).sum(-1)


    def predict_labels(self, distances):
        """
        Returns model predictions for multi-class and binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, np.int)
        
        for i in range(n_test):
            sorted_y = [y for _, y in sorted(zip(distances[i, :], self.train_y))]
            prediction[i] = max(set(sorted_y[:self.k]), key = sorted_y[:self.k].count)
        prediction = prediction.astype('str')
        return prediction
