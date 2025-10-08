"""
Hyperparameter Optimization for Classification Models Using Hyperopt

This module provides the `HyperoptSearch` class, which is designed as 
a wrapper for using the Hyperopt library. It supports hyperparameter
optimization searches over various classifiers from scikit-learn and 
allows evaluation using cross-validation metrics.

Classes:
    - HyperoptSearch: Handles hyperparameter tuning and model evaluation.

Example Usage:
    search = HyperoptSearch(X, y, cross_val=5, param_space=space, 
                            guide_metric='f1_overall', seed=42)
    search.begin_search(algorithm=tpe.suggest, search_length=100)
    best_params, results = search.get_results()
"""

import pandas as pd
import numpy as np
import time
from hyperopt import (
    STATUS_OK, fmin, Trials, space_eval, hp
)
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, f1_score, make_scorer, roc_auc_score
)
from sklearn.model_selection import (
    GroupKFold, LeaveOneGroupOut, cross_validate, cross_val_predict
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from utils import custom_cross_validate

class HyperoptSearch:
    
    def __init__(self, X, y, cross_val, param_space, guide_metric, groups=None, seed=None):
        """"
        Creates a Hyperopt Search Object
        
        Args:
            X (array-like): Feature Matrix for Task
                shape: (n_samples, feature_dimension)
                
            y (array-like): Target Values
                (shape: (n_samples, ))
                
            cross_val (int or cross-validation generator): Determines the 
                cross-validation splitting strategy.
                - `int`: Number of folds for `KFold`.
                - `KFold`, `StratifiedKFold`, `LeaveOneOut`, `LeavePOut`,
                `ShuffleSplit`, or `GroupKFold`: Uses the specified method.
                       
            param_space (Dict[str, Any]): The search space for hyperparameter optimization.
                - Defines the range of values to explore during optimization.
                - Should be a dictionary where keys are hyperparameter names, and values 
                  are Hyperopt search distributions.
                - Please visit section 2 of the Hyperopt documentation for more details
                  https://github.com/hyperopt/hyperopt/wiki/FMin
                    
            guide_metric (string): The evaluation metric used to guide the optimization process.
                - Options include: 'f1_overall', 'accuracy_overall', 'balanced_accuracy,
                  'kappa_overall', 'avg_f1', 'avg_kappa', 'avg_accuracy', 'roc_avg'.
                   See 'search_results' from 'get_results' for descriptions of each metric.
            
            groups (Optional[array-like]): Group labels for samples, required for 
                `GroupKFold` or similar strategies.
                - `None` (default): No grouping is applied.
                - `array-like (shape: (n_samples,))`: Each sample is assigned a 
                       group label, ensuring that samples from the same group are
                       not split across train and test sets.
            
            seed (Optional int): Ensure reproducibility 
        """
        
        self.X = X
        self.y = y
        self.cv = cross_val
        self.groups = groups
        self.param_space = param_space
        self.guide_metric = guide_metric
        self.rstate = np.random.default_rng(seed)
        
        self.best_params = None
        self.trials = None
        
        
    def begin_search(self, algorithm, search_length):
        """
        Perform a hyperopt search on the current object and store 
        the results in the object's instance variables
        
        Args:
            algorithm (string): Hyperopt algorithm to use, must
                correspond to an existing hyperopt algorithm 
                (tpe.suggest, etc)
                
            search_length (int): Maximum number of trials to run
                search for
        """
        trials = Trials()  
        best = fmin(self.__objective, space=self.param_space,
                    algo=algorithm, max_evals=search_length, 
                    trials=trials, rstate=self.rstate)
        
        self.best_params = space_eval(self.param_space, best)
        self.trials = trials
    
    def get_search_results(self):
        """
        Retrieves a dataframe of the results from the most recent model search,
        sorted by the guide metric.

        Returns:                
            - search_results (pandas.DataFrame): Result of the most recent 
            model search. Each row describes parameters and metrics of a 
            given model trial. The columns include:
                - `trial_id` (int): Unique identifier, incremented sequentially
                by Hyperopt for each trial.
                - `model_name` (str)
                - `params` (dict): A dictionary containing the hyperparameters 
                used for the model in the trial.
                - `avg_f1` (float): F1 score taken on the fold-level and averaged
                across folds.
                - `std_f1` (float): The standard deviation of the F1 score across
                folds.
                - `avg_kappa` (float): Cohen's Kappa score taken on the fold-level
                and averaged across folds.
                - `std_kappa` (float): The standard deviation of the kappa score
                across folds.
                - `avg_accuracy` (float): Accuracy taken on the fold-level and
                averaged across folds.
                - `std_accuracy` (float): The standard deviation of the accuracy
                across folds.
                - `f1_overall` (float): The overall F1 score for the model,
                calculated by comparing the predicted labels to the actual labels
                across all folds/trials combined.
                - `kappa_overall` (float): The overall kappa score for the model,
                calculated by comparing the predicted labels to the actual labels
                across all folds/trials combined.
                - `accuracy_overall` (float): The overall accuracy score for the model,
                calculated by comparing the predicted labels to the actual labels
                across all folds/trials combined.
                - `balanced_accuracy` (float): The balanced accuracy score of 
                the model.
                - `roc_auc_overall` (dict): Contains the ROC AUC scores for each class
                - `weighted_roc` (dict): Contains the weighted ROC AUC score for each class.
                - `roc_avg` (float): The average weighted ROC AUC score across all classes.
                - `confusion_matrix` (array-like)
                - `y_pred` (array-like): The predicted values from the model.
                - `y_actual` (array-like): The actual values used to compare 
                against predictions.
        Raises:
            RuntimeError: If no model search has been performed (i.e., `begin_search()` was not called).
        """
        if self.trials is None:
            raise RuntimeError(
                "No model search has been performed; "
                "call begin_search() before attempting to retrieve results."
            )

        search_results = pd.DataFrame([trial['model_metrics'] for trial in self.trials.results])
        search_results['trial_id'] = [trial['tid'] for trial in self.trials]
        
        # Sort results by guide_metric
        search_results = search_results.sort_values(by=self.guide_metric, ascending=False)
        search_results = search_results.reset_index(drop=True)
        
        columns = ["trial_id", "model_name", "params", "avg_f1", "std_f1", "avg_kappa", "std_kappa",
                    "avg_accuracy", "std_accuracy", "f1_overall", "kappa_overall",
                    "accuracy_overall", "balanced_accuracy", "roc_auc_overall",
                    "weighted_roc","roc_avg", "confusion_matrix", "y_pred",
                    "y_actual"]
        search_results = search_results[columns] # Reorder columns for convenience
        
        return search_results
        
    def get_grouped_results_by_trial(self):
        """
        Retrieves grouped evaluation metrics for each trial from the model search.

        Returns:
            list of dict: A list where each element corresponds to a trial and contains:
                - 'trial_id' (int): Unique identifier of the trial.
                - 'results_by_group' (pd.DataFrame): DataFrame of per-group metrics 
                as returned by `custom_cross_validate`, including group IDs, true and 
                predicted labels, and scores for each specified metric.

        Raises:
            RuntimeError: If no model search has been performed (i.e., `begin_search()` was not called).
        """
        if self.trials is None:
            raise RuntimeError(
                "No model search has been performed; "
                "call begin_search() before attempting to retrieve results."
            )
            
        trial_results = [
            {
                "trial_id": trial['tid'],
                "results_by_group": pd.DataFrame(result['scores_by_group'])
            }
            for trial, result in zip(self.trials.trials, self.trials.results)
        ]
        return trial_results

    def __get_classifier_instance(self, model_name, model_params):
        """
        Returns the correct classifier given it's parameters
        
        Args:
            model_params (dictionary): Parameters of model to create,
                matching the format of hyperopt's search space
        
        Returns:
            scikit-learn classifier object
        """
        model_seed = self.rstate.integers(0, 2**32)
        
        if model_name == 'naive_bayes':
            return GaussianNB(**model_params)
        elif model_name == 'AdaBoost':
            return AdaBoostClassifier(**model_params, 
                                      random_state=model_seed)
        elif model_name == 'LinSVC':
            return LinearSVC(**model_params, random_state=model_seed)      
        elif model_name == 'decision_tree':
            return DecisionTreeClassifier(**model_params, 
                                          random_state=model_seed) 
        elif model_name == 'randomforest':
            return RandomForestClassifier(**model_params, 
                                          random_state=model_seed) 
        elif model_name == 'knn':
            return KNeighborsClassifier(**model_params)
        elif model_name == 'logistic_regression':
            return LogisticRegression(**model_params,
                                      random_state=model_seed)
        elif model_name == 'SVC':
            return SVC(**model_params, probability=True, 
                       random_state=model_seed)
        elif model_name == 'SVM':
            return svm.SVC(**model_params, gamma='auto', probability=True, 
                           random_state=model_seed) 
        else:
            return 0
        
        
    def __compute_metrics(self, y_pred, y_proba):
        """
        Calculates overall metrics based on final predictions 
        and ground-truth values
        
        Args:
            y_pred(NumPy Array): Array of predicted class. Dimension: (n_samples,)
            
            y_proba(): Array of predicted probabilities for each class. 
                Dimension: (n_samples, n_classes)
        
        Returns:
            final_metrics (Dict) : consists of
            metric_name, metric value  key-value pairs  
        """
        
        final_metrics = {
            'f1_overall': f1_score(self.y, y_pred, average='weighted'),
            'accuracy_overall': accuracy_score(self.y, y_pred),
            'balanced_accuracy': balanced_accuracy_score(self.y, y_pred),
            'kappa_overall': cohen_kappa_score(self.y, y_pred),
            'confusion_matrix': confusion_matrix(self.y, y_pred)
        }
        
        roc_auc_ovr = {}
        weighted_roc = {}
        roc_avg = 0
        for class_id in range(2):
            prob_tmp = y_proba[:, class_id]
            true_max_tmp = [1 if y_tmp == class_id else 0 for y_tmp in self.y]

            try:
                roc_auc_ovr[class_id] = roc_auc_score(true_max_tmp, prob_tmp)
                weighted_roc[class_id] = roc_auc_score(true_max_tmp, prob_tmp, average='weighted')
                if(class_id > 0): roc_avg += weighted_roc[class_id]
            except:
                print(f"Issue calculating ROC with class {class_id}")
                roc_auc_ovr[class_id] = None
        
        final_metrics['roc_auc_overall'] = roc_auc_ovr
        final_metrics['weighted_roc'] = weighted_roc
        final_metrics['roc_avg'] = roc_avg
        final_metrics['y_pred'] = y_pred
        final_metrics['y_actual'] = self.y
        
        return final_metrics
    

    def __evaluate_model(self, classifier):
        """
        Get overall and cross-validation metrics for classifier 
        
        Args:
            classifier (sci-kit learn classifier object)

        Returns:
            all_metrics (Dict): consists of
                metric_name, metric value  key-value pairs
        """
        y_proba = cross_val_predict(classifier,self.X, self.y, groups=self.groups,
                            n_jobs=20,cv=self.cv, verbose=0, method='predict_proba')
        
        # Generate predictions from probabilities
        y_pred = [1 if prob[1] > 0.5 else 0 for prob in y_proba] 
        y_pred = np.array(y_pred, dtype=float)
        
        scoring = {
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
        'kappa': lambda y_true, y_pred: cohen_kappa_score(y_true, y_pred),
        'accuracy': lambda y_true, y_pred: accuracy_score(y_true, y_pred) 
        }
        
        scores_by_group = custom_cross_validate(self.y, y_pred, self.groups, scoring)
        
        cross_val_metrics = {
            'avg_f1': np.mean(scores_by_group['f1']),
            'std_f1': np.std(scores_by_group['f1']),
            'avg_kappa': np.mean(scores_by_group['kappa']),
            'std_kappa': np.std(scores_by_group['kappa']),
            'avg_accuracy': np.mean(scores_by_group['accuracy']),
            'std_accuracy': np.std(scores_by_group['accuracy'])
        }
        
        final_metrics = self.__compute_metrics(y_pred, y_proba)
        all_metrics = {**cross_val_metrics, **final_metrics}
        
        return all_metrics, scores_by_group
        
        
    def __objective(self, model_params):
        """
        Objective funtion to integrate with hyperopt's fmin() 
        
        Args:
            model_params (dictionary)

        Returns:
            (dictionary) : must include a loss metric 
                (to be minimized by Hyperopt)
        """
        
        model_name = model_params.pop('type')
        classifier = self.__get_classifier_instance(model_name, model_params)
        
        model_metrics, scores_by_group = self.__evaluate_model(classifier)
        
        model_metrics['model_name'] = model_name
        model_metrics['params'] = model_params
        
        loss = -(model_metrics[self.guide_metric])
        
        return {"loss": loss if not np.isnan(loss) else float("inf"), 
                'status': STATUS_OK,
                'eval_time': time.time(), 
                'model_metrics': model_metrics,
                'scores_by_group': scores_by_group
                }
