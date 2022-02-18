import numpy as np

#For fitting
from .utils import BaggingClassifier, stats, return_importance_scores
from .tree import MTree
import shap as sh
from mECFS import mECFS

#For class construction
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import balanced_accuracy_score

from abc import ABCMeta, abstractmethod
from gc import collect

class LANDMarkModel(BaseEstimator, metaclass=ABCMeta):
    """Base Class for all LANDMark Classification Models"""

    def fit(self, X, y):
        """Fits a LANDMark Model"""

        return self._fit(X, y)

    def predict(self, X):
        """Predict Using a LANDMark Model"""

        return self._predict(X)

    def predict_proba(self, X):
        """Return Class Probabilities Using a LANDMark Model"""

        return self._predict_proba(X)

    def score(self, X, y):
        """Returns the Balanced Accuracy Score"""

        return self._score(X)

    def proximity(self):
        """Returns the Tree Embedding"""

        return self._proximity(X)

    @abstractmethod
    def _fit(self, X, y):
        """Fits a LANDMark Model"""

    @abstractmethod
    def _predict(self, X):
        """Predict Using a LANDMark Model"""

    @abstractmethod
    def _predict_proba(self, X):
        """Return Class Probabilities Using a LANDMark Model"""

    @abstractmethod
    def _score(self, X, y):
        """Returns the Balanced Accuracy Score"""

    @abstractmethod
    def _proximity(self):
        """Returns the Tree Embedding"""

class LANDMarkClassifier(ClassifierMixin, LANDMarkModel):
 
    def __init__(self, 
                 n_estimators = 64,
                 min_samples_in_leaf = 5, 
                 max_depth = -1,
                 max_features = 0.80,
                 min_gain = 0,
                 impurity = "gain",
                 use_oracle = True,
                 bootstrap = False,
                 use_ifs = False,
                 n_feats_sel = 75,
                 use_xgbt = True,
                 n_jobs = 4):
        
        #Tree construction parameters
        self.n_estimators = n_estimators
        self.min_samples_in_leaf = min_samples_in_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_gain = min_gain
        self.impurity = impurity
        self.use_oracle = use_oracle
        self.bootstrap = bootstrap
        self.use_ifs = use_ifs
        self.n_feats_sel = n_feats_sel
        self.use_xgbt = use_xgbt

        self.n_jobs = n_jobs

    def _fit(self, X, y):
        """
        Parameters
        -----------
        X : Numpy array of samples with shape (n_samples, n_features)

        y : Numpy array of class labels with shape (n_samples,)

        Returns
        -----------
        self : object, the fitted model
        """
        self.classes_ = np.unique(y)

        self.n_classes_ = self.classes_.shape[0]
        
        #Fit a model       
        self.retained = np.asarray([True for i in range(X.shape[1])])
        F_idx_current = np.asarray([i for i in range(X.shape[1])])

        if self.use_ifs:
            self.retained = mECFS(n_init = 30, 
                                  k_select = self.n_feats_sel, 
                                  bootstrap = True, 
                                  use_xgbt = self.use_xgbt,
                                  n_jobs = self.n_jobs).fit(X, y).selected_

            F_idx_current = F_idx_current[self.retained]

        self.estimators_ = BaggingClassifier(base_estimator = MTree(min_samples_in_leaf = self.min_samples_in_leaf,
                                                                    max_depth = self.max_depth,
                                                                    max_features = self.max_features,
                                                                    min_gain = self.min_gain,
                                                                    impurity = self.impurity,
                                                                    use_oracle = self.use_oracle,
                                                                    bootstrap = self.bootstrap),
                                             n_estimators = self.n_estimators,
                                             n_jobs = self.n_jobs)

        self.estimators_.fit(X[:, self.retained], y)

        self.avg_depth = np.mean([estimator.max_depth for estimator in self.estimators_.estimators_])

        #Get feature importance scores
        self.feature_importances_ = np.zeros(shape = (X.shape[1],))

        feature_importances_ = return_importance_scores(self.estimators_.estimators_)

        for i, idx in enumerate(F_idx_current):
            self.feature_importances_[idx] = feature_importances_[i]

        collect()

        return self

    def _predict(self, X):
        """
        Parameters
        -----------
        X : Numpy array of samples with shape (n_samples, n_features)

        Returns
        -----------
        predictions : Numpy array of predictions with shape (n_samples,)
        """
        predictions = self.estimators_.predict(X[:, self.retained])

        collect()

        return predictions
 
    def _predict_proba(self, X):
        """
        Parameters
        -----------
        X : Numpy array of samples with shape (n_samples, n_features)

        Returns
        -----------
        predictions : Numpy array of probabilities with shape (n_samples, n_classes)
        """
        predictions = self.estimators_.predict_proba(X[:, self.retained])

        collect()

        return predictions

    def _score(self, X, y):

        score = balanced_accuracy_score(y,
                                        self.predict(X[:, self.retained]))

        collect()

        return score

    def _proximity(self, X):

        tree_mats = []

        for estimator in self.estimators_.estimators_:
            tree_mats.append(estimator.proximity(X[:, self.retained]))

        emb = np.hstack(tree_mats)

        collect()

        return emb

class StatisticalLANDMarkClassifier(ClassifierMixin, LANDMarkModel):
 
    def __init__(self, 
                 n_estimators = 64,
                 min_samples_in_leaf = 5, 
                 max_depth = -1,
                 max_features = 0.80,
                 min_gain = 0,
                 impurity = "gain",
                 use_oracle = True,
                 bootstrap = False,
                 use_ifs = True,
                 n_feats_sel = 75,
                 use_xgbt = True,
                 max_replicates = 3,
                 p_value_init = 0.2,
                 p_value_fin = 0.05,
                 n_jobs = 4):
        
        #Tree construction parameters
        self.n_estimators = n_estimators
        self.min_samples_in_leaf = min_samples_in_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_gain = min_gain
        self.impurity = impurity
        self.use_oracle = use_oracle
        self.bootstrap = bootstrap

        self.use_ifs = use_ifs
        self.n_feats_sel = n_feats_sel
        self.use_xgbt = use_xgbt
        self.max_replicates = max_replicates
        self.p_value_init = p_value_init
        self.p_value_fin = p_value_fin

        self.n_jobs = n_jobs

    def _fit(self, X, y):
        """
        Fits a LANDMark Model.

        Parameters
        -----------
        X : Numpy array of samples with shape (n_samples, n_features)

        y : Numpy array of class labels with shape (n_samples,)

        Returns
        -----------
        self : object, the fitted model
        """

        self.classes_ = np.unique(y)

        self.n_classes_ = self.classes_.shape[0]

        #Statistical feature selection: Fit a model, identify candidate features, remove uninformative features, repeat           
        F_idx_current = np.asarray([i for i in range(X.shape[1])])

        if self.use_ifs:
            self.retained = mECFS(n_init = 30, 
                                  k_select = self.n_feats_sel, 
                                  bootstrap = True, 
                                  use_xgbt = self.use_xgbt,
                                  n_jobs = self.n_jobs).fit(X, y).selected_

            F_idx_current = F_idx_current[self.retained]

        else:
            self.retained = np.asarray([True for _ in range(X.shape[1])])

        for i in range(2):
            scores_mu = None

            self.max_evals = 2 * X[:, self.retained].shape[1] + 1

            for replicate in range(self.max_replicates):
                #Fit model
                est_tmp = BaggingClassifier(base_estimator = MTree(min_samples_in_leaf = self.min_samples_in_leaf,
                                                                   max_depth = self.max_depth,
                                                                   max_features = self.max_features,
                                                                   min_gain = self.min_gain,
                                                                   impurity = self.impurity,
                                                                   use_oracle = self.use_oracle,
                                                                   bootstrap = self.bootstrap),
                                            n_estimators = self.n_estimators,
                                            n_jobs = 7)

                est_tmp.fit(X[:, self.retained], y)

                #Calculate Shapley Values
                explainer = sh.Explainer(est_tmp.predict_proba,
                                         sh.maskers.Independent(X[:, self.retained],
                                                                max_samples = 200))
                
                try:
                    scores = explainer(X[:, self.retained],
                                       max_evals = self.max_evals).values

                except:
                    scores = explainer(X[:, self.retained]).values

                if replicate == 0:
                    scores_mu = scores

                else:
                    scores_mu += scores

            scores = scores_mu / float(self.max_replicates)

            #Extract Shapley Values from each group - Need to make this for multiple groups
            retained_new = stats(scores, 
                                 y, 
                                 self.retained.sum(),
                                 self.n_classes_,
                                 self.p_value_init,
                                 self.p_value_fin,
                                 i)

            F_idx_current = F_idx_current[retained_new]
            F_new_index = set(F_idx_current)

            for j in range(self.retained.shape[0]):
                if j in F_new_index:
                    pass
                else:
                    self.retained[j] = False

        #Fit final model on features identified using final iteration
        self.estimators_ = BaggingClassifier(base_estimator = MTree(min_samples_in_leaf = self.min_samples_in_leaf,
                                                                    max_depth = self.max_depth,
                                                                    max_features = self.max_features,
                                                                    min_gain = self.min_gain,
                                                                    impurity = self.impurity,
                                                                    use_oracle = self.use_oracle,
                                                                    bootstrap = self.bootstrap),
                                             n_estimators = self.n_estimators,
                                             n_jobs = self.n_jobs)

        self.estimators_.fit(X[:, self.retained], y)

        #Calculate the average depth of the model
        self.avg_depth = np.mean([estimator.max_depth for estimator in self.estimators_.estimators_])
        
        #Get feature importance scores
        feature_importances = return_importance_scores(self.estimators_.estimators_)

        self.feature_importances_ = np.zeros(shape = (X.shape[1]))
        for i, loc in enumerate(F_idx_current):
            self.feature_importances_[loc] = feature_importances[i]

        collect()

        return self

    def _predict(self, X):
        """
        Predicts Class Labels using a LANDMark Model.

        Parameters
        -----------
        X : Numpy array of samples with shape (n_samples, n_features)

        Returns
        -----------
        predictions : Numpy array of predictions with shape (n_samples,)
        """
        predictions = self.estimators_.predict(X[:, self.retained])

        collect()

        return predictions
 
    def _predict_proba(self, X):
        """
        Returns Array of Class Probabilities.

        Parameters
        -----------
        X : Numpy array of samples with shape (n_samples, n_features)

        Returns
        -----------
        predictions : Numpy array of probabilities with shape (n_samples, n_classes)
        """
        predictions = self.estimators_.predict_proba(X[:, self.retained])

        collect()

        return predictions

    def _score(self, X, y):
        """
        Returns the Balanced Accuracy Score of a LANDMark Model.

        Parameters
        -----------
        X : Numpy array of samples with shape (n_samples, n_features)

        y : Numpy array of class labels with shape (n_samples,)

        Returns
        -----------
        score : Numpy array, Balanced Accuracy Scores
        """

        score = balanced_accuracy_score(y,
                                        self.predict(X[:, self.retained]))

        collect()

        return score

    def _proximity(self, X):
        """
        Returns the Tree Embedding of a LANDMark Model.

        Parameters
        -----------
        X : Numpy array of samples with shape (n_samples, n_features)

        Returns
        -----------
        emb : Numpy array, Embedding
        """

        tree_mats = []

        for estimator in self.estimators_.estimators_:
            tree_mats.append(estimator.proximity(X[:, self.retained]))

        emb = np.hstack(tree_mats)

        collect()

        return emb