import numpy as np
import pickle

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        ''' Konstruktor ''' 
        
        # Untuk node keputusan
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        
        # Untuk node daun
        self.value = value

class DecisionTreeRegressor():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' Konstruktor '''
        
        # Inisialisasi akar pohon
        self.root = None
        
        # Kondisi penghentian
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def _build_tree(self, dataset, curr_depth=0):
        ''' Fungsi rekursif untuk membangun pohon '''

        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        # Membagi data hingga kondisi penghentian terpenuhi
        if num_samples >= self.min_samples_split and (self.max_depth is None or curr_depth <= self.max_depth):
            # Mencari pemisahan terbaik
            best_split = self._get_best_split(dataset, num_samples, num_features)
            # Memeriksa apakah pemisahan valid
            if best_split and "var_red" in best_split and best_split["var_red"] > 0:
                # Rekursi ke kiri
                left_subtree = self._build_tree(best_split["dataset_left"], curr_depth+1)
                # Rekursi ke kanan
                right_subtree = self._build_tree(best_split["dataset_right"], curr_depth+1)
                # Mengembalikan node keputusan
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["var_red"])

        # Menghitung nilai untuk node daun
        leaf_value = self._calculate_leaf_value(Y)
        return Node(value=leaf_value)
    
    def _get_best_split(self, dataset, num_samples, num_features):
        ''' Fungsi untuk mencari pemisahan terbaik '''
        
        best_split = {}
        max_var_red = -float("inf")
        # Loop untuk setiap fitur
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # Loop untuk setiap nilai fitur dalam data
            for threshold in possible_thresholds:
                # Mendapatkan pemisahan saat ini
                dataset_left, dataset_right = self._split(dataset, feature_index, threshold)
                # Memeriksa apakah kedua cabang tidak kosong
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # Menghitung pengurangan varians
                    curr_var_red = self._variance_reduction(y, left_y, right_y)
                    # Memperbarui pemisahan terbaik jika diperlukan
                    if curr_var_red > max_var_red:
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "dataset_left": dataset_left,
                            "dataset_right": dataset_right,
                            "var_red": curr_var_red
                        }
                        max_var_red = curr_var_red
                        
        return best_split if best_split else None
    
    def _split(self, dataset, feature_index, threshold):
        ''' Fungsi untuk membagi data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right
    
    def _variance_reduction(self, parent, l_child, r_child):
        ''' Fungsi untuk menghitung pengurangan varians '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction
    
    def _calculate_leaf_value(self, Y):
        ''' Fungsi untuk menghitung nilai node daun '''
        return np.mean(Y)
                
    def print_tree(self, tree=None, indent=" "):
        ''' Fungsi untuk mencetak pohon keputusan '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.var_red)
            print("%sKiri:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sKanan:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' Fungsi untuk melatih model '''
        
        dataset = np.concatenate((X, Y.values.reshape(-1, 1)), axis=1)
        self.root = self._build_tree(dataset)
        
    def _make_prediction(self, x, tree):
        ''' Fungsi untuk memprediksi dataset baru '''
        
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._make_prediction(x, tree.left)
        else:
            return self._make_prediction(x, tree.right)
    
    def predict(self, X):
        ''' Fungsi untuk melakukan prediksi '''
        
        return [self._make_prediction(x, self.root) for x in X.values]
    
class RandomForestRegressor():
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, bootstrap=True, verbose=False):
        ''' Konstruktor '''
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.verbose = verbose
        self.trees = []

    def _bootstrap_sample(self, X, y):
        ''' Fungsi untuk mengambil sampel bootstrap '''
        n_samples = len(X)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X.iloc[indices], y.iloc[indices]

    def fit(self, X, y):
        ''' Fungsi untuk melatih model '''
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
                print(f"Tree {i + 1}/{self.n_estimators} telah dilatih.")

    def predict(self, X):
        ''' Fungsi untuk melakukan prediksi '''
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return predictions.mean(axis=0)
    
    def save_model(self, filename):
        ''' Menyimpan model ke file menggunakan pickle '''
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load_model(filename):
        ''' Memuat model dari file menggunakan pickle '''
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model