import numpy as np

class VotingClassifier(object):
    """Class for executing voting classifier
    
    Usage example:
        >>> model1 = rda()
        >>> model2 = knn()
        >>> ensemble = VotingClassifier(models=[{'rda':model1},{'knn':model2}], voting='soft')
        >>> ensemble.fit(train_x,train_y)
        >>> ensemble.predict(test_x)
    """
    def __init__(self, models, voting, weights=None):
        """
        Args:
            models: Models using for ensemble, this parameter is taken by dictionary
            Voting: Type for voting. It has two choice, soft and hard
            weights: Weight given to probability of models
        Attributes:
            models: Models using for ensemble
            voting: Voting type
            named_estimator: Estimator mapped to each name
            estimator: list of estimators with no name matched
            weight: weight given for each probabilty of each model has returned
        """
        if voting not in ['soft', 'hard']:
            raise NameError('Should Choose one of ["soft", "hard"]')

        else:
            self.voting = voting
        
        self.named_estimator = models
        self.estimator = [list(m.values())[0] for m in models]
        self.weights = weights

    def _softmax(self,x):
        """ Applies softmax to an input x"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def fit(self, x, y):
        """ Train each model
        
        Args:
            X: The feature of the dataset
            y: The class of the dataset
        
        Returns:
            Self: The updated values of each model
        """
        self.class_names = np.unique(y)
        for model in self.estimator:
            model.fit(x,y)

    def _soft_predict(self, x):
        """Soft vote for the model probability
        
        Args:
            X: The feature of the dataset

        Returns:
            Class name of the maximum probablitiy 
        """
        pred = []
        for data in x:
            results = []
            data = data.reshape(1,-1)
            for model in self.estimator:
                results.append(model.predict_proba(data)[0])


            if self.weights:
                self.weights = [i/sum(self.weights) for i in self.weights]
                weighted_results = []
                for weight, probs in zip(self.weights, results):
                     weighted_results.append(np.asarray(probs) * weight)
                
                results = weighted_results

            weighted_mean = list(np.mean(np.asarray(results), axis=0))

            final_prob = self._softmax(weighted_mean)
            class_probs = {}
            
            for i, class_prob in zip(self.class_names, final_prob):
                class_probs[i] = class_prob

            pred.append(max(class_probs, key=class_probs.get))

        return pred

    def _hard_predict(self, x):
        """Hard vote for the model probability
        
        Args:
            X: The feature of the dataset

        Returns:
            Class name of the maximum probablitiy 
        """
        pred = []

        for data in x:
            results = []
            data = data.reshape(1,-1)
            for model in self.estimator:
                results.append(model.predict(data))

            unique, counts = np.unique(results, return_counts=True)
            class_count = dict(zip(unique, counts))

            pred.append(max(class_count, key=class_count.get))

        return pred


    def predict(self, x):
        """Result of ensemble model
        
        Args:
            X: The feature of the dataset

        Returns:
            Class name of the maximum probablitiy 
        """
        if self.voting =='hard':
            prediction = self._hard_predict(x)

        elif self.voting == 'soft':
            prediction = self._soft_predict(x)

        else:
            raise NameError('Should Choose one of ["soft", "hard"]')

        return prediction

    
    def _soft_predict_proba(self,x):
        """Predicting the class probability of the data via soft voting

        Calculate the problems for each class and return probability of each class.

        Args:
            x: feature data to predict class

        Returns:
            Probablitiy of each class 
        """
        pred_prob = []
        for data in x:
            results = []
            data = data.reshape(1,-1)
            for model in self.estimator:
                results.append(model.predict_proba(data)[0])


            if self.weights:
                self.weights = [i/sum(self.weights) for i in self.weights]
                weighted_results = []
                for weight, probs in zip(self.weights, results):
                     weighted_results.append(np.asarray(probs) * weight)
                
                results = weighted_results

            weighted_mean = list(np.mean(np.asarray(results), axis=0))

            final_prob = self._softmax(weighted_mean)
            
            pred_prob.append(final_prob)

        return np.asarray(pred_prob)

    
    def _hard_predict_proba(self,x):
        """Predicting the class probability of the data via hard voting

        Calculate the problems for each class and return probability of each class.

        Args:
            x: feature data to predict class

        Returns:
            Probablitiy of each class 
        """
        pred_prob = []

        for data in x:
            results = []
            data = data.reshape(1,-1)
            for model in self.estimator:
                results.append(model.predict(data))

            unique, counts = np.unique(results, return_counts=True)
            class_count = dict(zip(unique, counts))

            class_prob = []

            for count in class_count.values():
                class_prob.append(count / sum(class_count.values()))

            pred_prob.append(class_prob)
            

        return np.asarray(pred_prob)

    def predict_proba(self, x):
        """Result of ensemble model
        
        Args:
            X: The feature of the dataset

        Returns:
            Probablitiy of each class
        """
        if self.voting =='hard':
            prediction = self._hard_predict_proba(x)

        else:
            prediction = self._soft_predict_proba(x)

        return prediction
