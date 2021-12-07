import numpy as np

class DiscriminentAnalysis():
    """Class for executing RDA
    
    alpha=beta=0: quadric classifier
    alpha=0 and beta=1: shared covariance, linear classifier
    alpha=1 and beta =0: diagonal covariance, nearest mean classifier
    
    In Between theses extreme cases optimizing alpha and beta 
    
    Usage example:
        >>> dis = DiscriminentAnalysis(alpha=0.5, beta=0.7, eval_mode = False)
        >>> dis.fit(train_x, train_y)
        >>> dis.predict(test_x[0])
    """
    def __init__(self, alpha=0.0, beta=0.0):
        """
        Args:
            alpha: Tuning parameter alpha
            beta: Tuning parameter beta
        Attributes:
            learned: Keeps track of if RDA has been fit
            alpha: Regularization parameter in range [0, 1]
            beta: Regularization parameter in range [0, 1]
            class_names: Array of class names. [0, 1] for example.
            class_priors: Prior probability of each class.
            class_means: Vector means of each class
            regularized_covariances: Regulized covariances to calculate RDA covariance.
            rda_covariance: RDA covariance
            feature_dimension: The dimension of the feature data
        """
        self.learned = False
        self.alpha = alpha
        self.beta = beta
        self.class_names = []
        self.class_priors = {}
        self.class_means = {}
        self.regularized_covariances = {}
        self.rda_covariances = {}
        self.feature_dimension = 0
        self._estimator_type = "classifier"


    def reset(self, alpha, beta):
        """Resetting the parametes to initalized state."""
        self.learned = False
        self.class_names = []
        self.alpha = alpha
        self.beta = beta
        self.class_priors = {}
        self.class_means = {}
        self.regularized_covariances = {}
        self.rda_covariances = {}
        self.feature_dimension = 0


    def return_parameters(self):
        """Return the calculated paremters of the model"""
        parameters = {
            'alpha': self.alpha,
            'beta': self.beta,
            'class_name': self.class_names,
            'class_priors': self.class_priors,
            'class_means': self.class_means,
            'reg_cov': self.regularized_covariances,
            'rda_cov': self.rda_covariances
        }

        return parameters

    def fit(self, X, y):
        """Training function. Calculates the rda covariance.

        Changes the value of self.learned, so can use the predict function.
        
        Args:
            X: The feature of the dataset.
            y: The class of the dataset.

        Returns:
            Self: The updated values of the model.
        """
        self.class_names = np.unique(y)
        class_covariances = {}
        pooled_covariances = 0
        self.feature_dimension = X.shape[1]
        for i in self.class_names:
            class_indices = np.where(y == i)[0]
            class_samples = X[class_indices, :]
            self.class_priors[i] = float(len(class_indices)) / len(y)
            self.class_means[i] = np.mean(class_samples, axis=0)
            class_covariances[i] = np.cov(class_samples, rowvar=0)
            pooled_covariances += class_covariances[i] * self.class_priors[i]
        # Calculate regularized covariance matricies for each class
        for i in self.class_names:
            self.regularized_covariances[i] = (self.beta * pooled_covariances) + ((1 - self.beta) *class_covariances[i])

        # Calulate the RDA covarinace matriceis for each class
        for i in self.class_names:
            self.rda_covariances[i] = ((1-self.alpha) * self.regularized_covariances[i]) + (self.alpha * (1/self.feature_dimension) * np.trace(self.regularized_covariances[i]) * np.eye(self.regularized_covariances[i].shape[0]))
        
        # Changing the value to use the predict function
        self.learned = True
        return self

    def predict(self, x):
        """Predicting the class of the data

        Calculate the problems for each class and return the maximum value for the classes and return the class name

        Args:
            x: feature data to predict class

        Returns:
            Class name of the maximum probablitiy 
        """
        if not self.learned:
            raise NameError('Fit model first')
        # Determine probability of each class given input vector

        if len(x.shape) > 1:

            result = []

            for data in x:


                class_prob = {}
                for i in self.class_names:
                    # Divid the class delta calculation into 3 parts
                    part1 = -0.5 * np.log1p(np.linalg.det(self.rda_covariances[i]))
                    part2 = -0.5 * np.matmul(np.matmul((data - self.class_means[i]).T, np.linalg.pinv(self.rda_covariances[i])), (data - self.class_means[i]))
                    part3 = np.log(self.class_priors[i])
                    class_prob[i] = part1 + part2 + part3


                class_probs = self._softmax(list(class_prob.values()))

                for prob, k in zip(class_probs, class_prob.keys()):
                    class_prob[k] = prob

                
                result.append(max(class_prob, key=class_prob.get))

            return result


        else: 
            class_prob = {}
            for i in self.class_names:
                # Divid the class delta calculation into 3 parts
                part1 = -0.5 * np.log1p(np.linalg.det(self.rda_covariances[i]))
                part2 = -0.5 * np.matmul(np.matmul((x - self.class_means[i]).T, np.linalg.pinv(self.rda_covariances[i])), (x - self.class_means[i]))
                part3 = np.log(self.class_priors[i])
                class_prob[i] = part1 + part2 + part3


            class_probs = self._softmax(list(class_prob.values()))

            for prob, k in zip(class_probs, class_prob.keys()):
                class_prob[k] = prob

            
            
            return max(class_prob, key=class_prob.get)


    def predict_proba(self, x):
        """Predicting the class probability of the data

        Calculate the problems for each class and return probability of each class.

        Args:
            x: feature data to predict class

        Returns:
            Probablitiy of each class 
        """

        if not self.learned:
            raise NameError('Fit model first')
        # Determine probability of each class given input vector

        if len(x.shape) > 1:
            result = []

            for data in x:
                class_prob = {}
                for i in self.class_names:
                    # Divid the class delta calculation into 3 parts
                    part1 = -0.5 * np.log(np.linalg.det(self.rda_covariances[i]))
                    part2 = -0.5 * np.matmul(np.matmul((data - self.class_means[i]).T, np.linalg.pinv(self.rda_covariances[i])), (data - self.class_means[i]))
                    part3 = np.log(self.class_priors[i])
                    class_prob[i] = part1 + part2 + part3


                class_probs = self._softmax(list(class_prob.values()))

                result.append(class_probs)

            return np.asarray(result)

        
        else:
            class_prob = {}
            for i in self.class_names:
                # Divid the class delta calculation into 3 parts
                part1 = -0.5 * np.log1p(np.linalg.det(self.rda_covariances[i]))
                part2 = -0.5 * np.matmul(np.matmul((x - self.class_means[i]).T, np.linalg.pinv(self.rda_covariances[i])), (x - self.class_means[i]))
                part3 = np.log(self.class_priors[i])
                class_prob[i] = part1 + part2 + part3


            class_probs = self._softmax(list(class_prob.values()))

            return [class_probs]


    def fit_predict(self, train_x, train_y, test_x):
        self.fit(train_x, train_y)
        pred = self.predict(test_x)

        return np.asarray(pred)



    def _softmax(self,x):
        """ applies softmax to an input x"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()