import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class CustomDecisionTreeRegressor:

    # Initialising parameters
    def __init__(self, max_depth, min_samples_leafs):
        self.max_depth = max_depth
        self.min_samples_leafs = min_samples_leafs
        self.tree_=None

    # Finding the best split
    def best_split(self, X, y):

        n_samples, n_features = X.shape
        best_feature, best_threshold = None, None
        best_rss = float("inf")

        # Loop through every feature
        for f in range(n_features):
            # Getting values for that specific feature
            feature_values = X[:, f]

            # argsort returns the indexes of values in a sorted order
            sorted_idx = np.argsort(feature_values)

            # Getting both feature values and target values sorted
            fv_sorted = feature_values[sorted_idx]
            y_sorted = y[sorted_idx]

            # Defining total sums for later use. This is just the right region to start with
            total_count = n_samples
            total_sum = np.sum(y_sorted)
            total_sq_sum = np.sum(y_sorted ** 2)

            # Sums and counts for left region to start with
            left_count = 0
            left_sum = 0.0
            left_sq_sum = 0.0

            # looping through all the samples minus 1
            for i in range(n_samples - 1):
                yi = y_sorted[i]

                # Updating the left region variables
                left_count += 1
                left_sum += yi
                left_sq_sum += yi * yi

                # Updating right region
                right_count = total_count - left_count

                # There is no samples in the right region it breaks the loop
                if right_count == 0:
                    break
                
                # Continuing updating right region
                right_sum = total_sum - left_sum
                right_sq_sum = total_sq_sum - left_sq_sum

                # If the next feature value is the same as this one we go to next iteration
                if fv_sorted[i] == fv_sorted[i + 1]:
                    continue

                # Compute RSS for left and right using alternative variance formula for faster computation
                rss_left = left_sq_sum - (left_sum ** 2) / left_count
                rss_right = right_sq_sum - (right_sum ** 2) / right_count
                total_rss = rss_left + rss_right

                # Check if the error is smaller than the best rss until now
                if total_rss < best_rss:
                    best_rss = total_rss
                    best_feature = f
                    best_threshold = (fv_sorted[i] + fv_sorted[i + 1]) / 2

        return best_feature, best_threshold
    
    # Creating fit function to also make it compatiable with sklearn
    def fit(self, X,y):
        self.tree_ = self.build_tree(X, y, depth=0)
        return self

    # Recursive Tree Builder 
    def build_tree(self, X, y, depth=0):
        
        # if current depth is bigger or equal to max_depth. | If all the y-values are the same in this node. | if the number of samples in node is smaller than min_samples_leafs.
        # Then return mean of all the y-values
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_leafs:
            return Node(value=np.mean(y))

        # Get feature and threshold for node
        feature, threshold = self.best_split(X, y)

        # If there is no best split
        if feature is None:
            return Node(value=np.mean(y))
        
        # Creating mask to get the correct datapoint for both regions
        left_idx = X[:, feature] < threshold
        right_idx = X[:, feature] >= threshold

        # Continue building tree just in our new regions.
        left = self.build_tree(X[left_idx], y[left_idx], depth+1)
        right = self.build_tree(X[right_idx], y[right_idx], depth+1)

        return Node(feature, threshold, left, right)

    # Prediction 
    def predict_one(self, node, x):
        # If node is a leaf return the nodes value
        if node.value is not None:
            return node.value
        # Depending on the datapoints value for a specific feature it will either move left or right.
        if x[node.feature] < node.threshold:
            return self.predict_one(node.left, x)
        else:
            return self.predict_one(node.right, x)

    # For every data point in the array we call the predict_one function
    def predict(self, X):
        return np.array([self.predict_one(self.tree_, x) for x in X])