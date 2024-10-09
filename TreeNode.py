class TreeNode():
    def __init__(self, gini, gain=0, c_label=None, split_feature=None, split_value=None):
        """
        Class to initialize a TreeNode. 

        Args:
            gini: Gini index of the node. 
            gain: Gini gain (impurity reduction) from the split.
            class_label: Majority class label - only used for the leaf nodes.
            split_feature: Index of the feature to split on (if not leaf node)
            split_value: Threshold value to split on (if not lead node)
        """
        # Gini index to determine quality of the split
        self.gini = gini

        # Gini gain for the node
        self.gain = gain

        # Class label - majority class label which is only used for leaf node
        self.c_label = c_label

        # Feature index and threshold value for splitting - if node is not a leaf node
        self.split_feature = split_feature
        self.split_value = split_value

        # Left and right child nodes for splitting
        self.left = None
        self.right = None

        # Flag to indicate if node is a leaf node
        self.is_leaf = c_label is not None

    def is_terminal(self):
        """
        Checks whether node is a leaf node
        
        Returns: 
            True is node is a leaf node, False otherwise
        """
        return self.is_leaf


