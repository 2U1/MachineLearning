import numpy as np
import copy
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Metric.metrics import accuracy_score, precision_score, recall_score, f1_score

class GridSearchVoting(object):
    """Class for Gridsearch


    Finding the optimal weight for Ensembel model VotingClassifier

    Args:
        param_grid: Dictionary type that contains the list of hyper parameters to calculate

    Usage example:
        >>> model = VotingClassifier()
        >>> params = {
                        [[1,1,1], [1,2,1], [2,2,1]]
                    }
        >>> grid = GridSearchRDA(model, params)
        >>> grid.fit(x,y)
    """
    def __init__(self, model, param_grid, metric='accuracy'):
        """
        Args:
            model: The VotingClassifier model to train.
            param_grid: The list of paramters in the dictionary.
            metric: metric type from one of ['accuracy', 'f1-score', 'recall', 'precision']. The default type is accuracy 
        Attributes:
            model: The VotingClassifier model to train.
            param_grid: The list of paramters in the dictionary.
            best_weight: The best weight for the models in VotingClassifier
            metric: metric type from one of ['accuracy', 'f1-score', 'recall', 'precision']. The default type is accuracy 
            best_score: The best score that is selected
        """

        if metric not in ['accuracy', 'f1-score', 'recall', 'precision']:
            raise NameError('Need to be selected from [accuracy, f1-score, recall, precision]')
        
        self.model = model
        self.param_grid = param_grid
        self.best_score = 0
        self.best_weight = None
        self.metric = metric


    def fit(self, x, y, cv=3):
        """ The Training function using cross validation

        Using K-fold the function rotate from the list of weight to find the best weight
        
        Args:
            X: the feature data
            y: class label
            cv: The number of folds for K-fold
        """

        data_length = len(x)
        
        weight_list = self.param_grid['weights']
        
        if data_length % cv == 0:
            cv_x = np.split(x, cv)
            cv_y = np.split(y, cv)
            
        else:
            remain = data_length % cv
            cv_x = np.split(x[:-remain], cv)
            cv_y = np.split(y[:-remain], cv)


        for weight in weight_list:
            model = copy.deepcopy(self.model)
            accuracy_score_list = []
            recall_score_list = []
            precision_score_list = []
            f1_score_list = []
            
            for i in range(cv):
                model.weights = weight

                test_x_cv = cv_x[i]
                train_x_cv = np.vstack(cv_x[:i] + cv_x[i + 1:])

                test_y_cv = cv_y[i]
                train_y_cv = np.vstack(cv_y[:i] + cv_y[i + 1:]).flatten()

                
                model.fit(train_x_cv, train_y_cv)
                
                pred = model.predict(test_x_cv)


                accuracy_score_list.append(accuracy_score(test_y_cv, pred))
                recall_score_list.append(recall_score(test_y_cv, pred, zero_division=1))
                precision_score_list.append(precision_score(test_y_cv, pred, zero_division=1))
                f1_score_list.append(f1_score(test_y_cv, pred, zero_division=1))
            

            accuracy_mean_score = np.mean(np.array(accuracy_score_list))
            recall_mean_score = np.mean(np.array(recall_score_list))
            precision_mean_score = np.mean(np.array(precision_score_list))
            f1_mean_score = np.mean(np.array(f1_score_list))

            if self.metric == "f1-score":
                if f1_mean_score > self.best_score:
                    self.best_score = f1_mean_score
                    self.best_estimator = model
                    self.best_weight = weight

            elif self.metric == "accuracy":
                if accuracy_mean_score > self.best_score:
                    self.best_score = accuracy_mean_score
                    self.best_estimator = model
                    self.best_weight = weight
            
            elif self.metric == "recall":
                if recall_mean_score > self.best_score:
                    self.best_score = recall_mean_score
                    self.best_estimator = model
                    self.best_weight = weight

            elif self.metric == "precision":
                if precision_mean_score > self.best_score:
                    self.best_score = precision_mean_score
                    self.best_estimator = model
                    self.best_weight = weight

        return self