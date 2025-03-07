import numpy as np
import pickle

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        
        # for leaf node
        self.value = value

class DecisionTreeRegressor():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def _build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''

        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        # split until stopping conditions are met
        if num_samples >= self.min_samples_split and (self.max_depth is None or curr_depth <= self.max_depth):
            # find the best split
            best_split = self._get_best_split(dataset, num_samples, num_features)
            # check if a valid split exists
            if best_split and "var_red" in best_split and best_split["var_red"] > 0:
                # recur left
                left_subtree = self._build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self._build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["var_red"])

        # compute leaf node
        leaf_value = self._calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def _get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_var_red = -float("inf")
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self._split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_var_red = self._variance_reduction(y, left_y, right_y)
                    # update the best split if needed
                    if curr_var_red > max_var_red:
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "dataset_left": dataset_left,
                            "dataset_right": dataset_right,
                            "var_red": curr_var_red
                        }
                        max_var_red = curr_var_red
                        
        # return best split
        return best_split if best_split else None
    
    def _split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def _variance_reduction(self, parent, l_child, r_child):
        ''' function to compute variance reduction '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction
    
    def _calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''

        return np.mean(Y)
                
    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y.values.reshape(-1, 1)), axis=1)
        self.root = self._build_tree(dataset)
        
    def _make_prediction(self, x, tree):
        ''' function to predict new dataset '''
        
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._make_prediction(x, tree.left)
        else:
            return self._make_prediction(x, tree.right)
    
    def predict(self, X):
        ''' function to predict a single data point '''
        
        return [self._make_prediction(x, self.root) for x in X.values]
    
class RandomForestRegressor():
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, bootstrap=True, verbose=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.verbose = verbose
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = len(X)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X.iloc[indices], y.iloc[indices]

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            if self.verbose:
                print(f"Tree {i + 1}/{self.n_estimators} trained.")

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return predictions.mean(axis=0)
    
    def save_model(self, filename):
        ''' Save the model to a file using pickle '''
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load_model(filename):
        ''' Load the model from a file using pickle '''
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
