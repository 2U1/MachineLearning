import numpy as np
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Metric.metrics import accuracy_score, precision_score, recall_score, f1_score
import copy
import itertools
import multiprocessing

class GridSearchRDA():
    """Class for Gridsearch


    Finding the optimal alhpa and beta for the RDA model, and save the calculated parameters and the model itself.

    Args:
        param_grid: Dictionary type that contains the list of hyper parameters to calculate

    Usage example:
        >>> model = DiscriminentAnalysis()
        >>> params = {
                    'alpha': np.linspace(0.0, 1.0, num=11, endpoint=True),
                    'beta': np.linspace(0.0, 1.0, num=11, endpoint=True)
                    }
        >>> grid = GridSearchRDA(model, params)
        >>> grid.fit(x,y)

    """
    def __init__(self, model, param_grid, metric="accuracy"):
        """
        Args:
            model: The RDA model to train.
            param_grid: The list of paramters in the dictionary.
            metric: metric type from one of ['accuracy', 'f1-score', 'recall', 'precision']. The default type is accuracy 
        Attributes:
            model: The RDA model to train.
            param_grid: The list of paramters in the dictionary.
            alpha: Tuning parameter alpha
            beta: Tuning parameter alpha
            best_covaraince: Covariances when the model gets highest score
            best_score: The best score that is selected
        """

        if metric not in ['accuracy', 'f1-score', 'recall', 'precision']:
            raise NameError('Need to be selected from [accuracy, f1-score, recall, precision]')
        
        self.model = model
        self.param_grid = param_grid
        self.alpha = 0
        self.beta = 0
        self.best_score = 0
        self.metric = metric

    def _pararell_compute(self, data):
        
        paramlist = data[0]

        cv_x = data[1]
        cv_y = data[2]
        
        alpha = paramlist[0]
        beta = paramlist[1]
        
        accuracy_score_list = []
        recall_score_list = []
        precision_score_list = []
        f1_score_list = []
        
        for i in range(len(cv_x)):
            self.model.reset(alpha, beta)

            test_x_cv = cv_x[i]
            train_x_cv = np.vstack(cv_x[:i] + cv_x[i + 1:])

            test_y_cv = cv_y[i]
            train_y_cv = np.vstack(cv_y[:i] + cv_y[i + 1:]).flatten()

            
            pred = self.model.fit_predict(train_x_cv, train_y_cv, test_x_cv)
            
            # self.model.fit(train_x_cv, train_y_cv)


            # for data in test_x_cv:
            #     pred.append(self.model.predict(data))

            
            accuracy_score_list.append(accuracy_score(test_y_cv, pred))
            recall_score_list.append(recall_score(test_y_cv, pred))
            precision_score_list.append(precision_score(test_y_cv, pred, zero_division=1))
            f1_score_list.append(f1_score(test_y_cv, pred))
        
        accuracy_mean_score = np.mean(np.array(accuracy_score_list))
        recall_mean_score = np.mean(np.array(recall_score_list))
        precision_mean_score = np.mean(np.array(precision_score_list))
        f1_mean_score = np.mean(np.array(f1_score_list))

        # print("alpha:{0:.1f}, beta:{1:.1f}, accuracy:{2:.4f}, recall:{3:.4f}, precision:{4:.4f} ,f1-score:{5:.4f}"\
        #     .format(alpha, beta, accuracy_mean_score ,recall_mean_score ,precision_mean_score ,f1_mean_score))

        result = {}

        if self.metric == "f1-score":
            result['alpha'] = alpha
            result['beta'] = beta
            result['best_score'] = f1_mean_score
            result['best_estimator'] = copy.deepcopy(self.model)

        elif self.metric == "accuracy":
            result['alpha'] = alpha
            result['beta'] = beta
            result['best_score'] = accuracy_mean_score
            result['best_estimator'] = copy.deepcopy(self.model)
        
        elif self.metric == "recall":
            result['alpha'] = alpha
            result['beta'] = beta
            result['best_score'] = recall_mean_score
            result['best_estimator'] = copy.deepcopy(self.model)
        
        elif self.metric == "precision":
            result['alpha'] = alpha
            result['beta'] = beta
            result['best_score'] = precision_mean_score
            result['best_estimator'] = copy.deepcopy(self.model)

        return result


    def fit(self, X, y, cv=2):
        """ The Training function using cross validation

        Using K-fold the function rotate from the list of alpha and beta to find the best combination.
        
        Args:
            X: the feature data
            y: class label
            cv: The number of folds for K-fold
        """

        data_length = len(X)
        
        alpha_list = self.param_grid['alpha']
        beta_list = self.param_grid['beta']
        
        if data_length % cv == 0:
            cv_x = np.split(X, cv)
            cv_y = np.split(y, cv)
            
        else:
            remain = data_length % cv
            cv_x = np.split(X[:-remain], cv)
            cv_y = np.split(y[:-remain], cv)
        
        paramlist = list(itertools.product(alpha_list, beta_list))
        data_to_hand = (paramlist, cv_x, cv_y)

        results = multiprocessing.Pool(4).map(self._pararell_compute, data_to_hand)
        outputs = [results[0] for result in results]

        max_score_item = max(outputs, key=lambda x:x['best_score'])

        self.best_estimator = max_score_item['best_estimator']
        self.alpha = max_score_item['alpha']
        self.beta = max_score_item['beta']
        self.best_score = max_score_item['best_score']

        # for alpha in alpha_list:
        #     for beta in beta_list:
        #         accuracy_score_list = []
        #         recall_score_list = []
        #         precision_score_list = []
        #         f1_score_list = []
        #         for i in range(cv):
        #             self.model.reset(alpha, beta)

        #             test_x_cv = cv_x[i]
        #             train_x_cv = np.vstack(cv_x[:i] + cv_x[i + 1:])

        #             test_y_cv = cv_y[i]
        #             train_y_cv = np.vstack(cv_y[:i] + cv_y[i + 1:]).flatten()

                    
        #             pred = self.model.fit_predict(train_x_cv, train_y_cv, test_x_cv)
                    
        #             # self.model.fit(train_x_cv, train_y_cv)


        #             # for data in test_x_cv:
        #             #     pred.append(self.model.predict(data))

                    
        #             accuracy_score_list.append(accuracy_score(test_y_cv, pred))
        #             recall_score_list.append(recall_score(test_y_cv, pred))
        #             precision_score_list.append(precision_score(test_y_cv, pred, zero_division=1))
        #             f1_score_list.append(f1_score(test_y_cv, pred))
                
        #         accuracy_mean_score = np.mean(np.array(accuracy_score_list))
        #         recall_mean_score = np.mean(np.array(recall_score_list))
        #         precision_mean_score = np.mean(np.array(precision_score_list))
        #         f1_mean_score = np.mean(np.array(f1_score_list))

        #         print("alpha:{0:.1f}, beta:{1:.1f}, accuracy:{2:.4f}, recall:{3:.4f}, precision:{4:.4f} ,f1-score:{5:.4f}"\
        #             .format(alpha, beta, accuracy_mean_score ,recall_mean_score ,precision_mean_score ,f1_mean_score))

                
        #         if self.metric == "f1-score":
        #             if f1_mean_score > self.best_score:
        #                 self.best_score = f1_mean_score
        #                 self.alpha = alpha
        #                 self.beta = beta
        #                 self.best_estimator = copy.deepcopy(self.model)

        #         elif self.metric == "accuracy":
        #             if accuracy_mean_score > self.best_score:
        #                 self.best_score = accuracy_mean_score
        #                 self.alpha = alpha
        #                 self.beta = beta
        #                 self.best_estimator = copy.deepcopy(self.model)
                
        #         elif self.metric == "recall":
        #             if recall_mean_score > self.best_score:
        #                 self.best_score = recall_mean_score
        #                 self.alpha = alpha
        #                 self.beta = beta
        #                 self.best_estimator = copy.deepcopy(self.model)
                
        #         elif self.metric == "precision":
        #             if precision_mean_score > self.best_score:
        #                 self.best_score = precision_mean_score
        #                 self.alpha = alpha
        #                 self.beta = beta
        #                 self.best_estimator = copy.deepcopy(self.model)

        return self
