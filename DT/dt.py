import numpy as np
import pandas as pd

class DT:
    def __init__(self,max_depth= None):
        self.max_depth = max_depth
        self.tree = None

    def gini_imp(self,y):
        classes,counts = np.unique(y,return_counts=True)
        p = counts/len(y)
        return 1-(np.sum(p**2))
    
    def split(self,X_colmn,threshold):
        left = np.where(X_colmn <= threshold)[0]
        right = np.where(X_colmn > threshold)[0]
        return left,right
    
    def gini_split(self,y,left,right):
        n = len(y)
        n_left = len(left)
        n_right = len(right)

        left_gini = self.gini_imp(y[left])
        right_gini = self.gini_imp(y[right])

        if n_left == 0 or n_right == 0:
            return float('inf')

        return (n_left*left_gini)/n + (n_right*right_gini)/n
    
    def best_split(self,X,y):
        best_threshold= None
        best_feature = None
        best_gini = float('inf')

        for i in range(X.shape[1]):
            X_colum = X[:,i]
            thresholds = np.unique(X_colum)
            for threshold in thresholds:
                left,right = self.split(X_colum,threshold)
                gini = self.gini_split(y,left,right)
                if gini< best_gini:
                    best_gini = gini
                    best_feature = i
                    best_threshold = threshold
        return best_feature,best_threshold,best_gini
    
    def build_tree(self,X,y,depth = 0):
        if len(np.unique(y))== 1 or (self.max_depth is not None and depth>= self.max_depth):
            return np.argmax(np.bincount(y))
        
        feature,threshold,gini = self.best_split(X,y)
        left,right = self.split(X[:,feature],threshold)
        left_tree = self.build_tree(X[left],y[left],depth+1)
        right_tree = self.build_tree(X[right],y[right],depth+1)

        if gini == float('inf'):
            return np.argmax(np.bincount(y))

        return {
            "feature": feature,
            "threshold":threshold,
            "left":left_tree,
            "right":right_tree
        }
    def fit(self,X,y):
        self.tree= self.build_tree(X,y)

    def predict_sample(self,x,tree):
        if isinstance(tree,dict):
            feature = tree["feature"]
            threshold = tree["threshold"]
            if x[feature] <= threshold:
                return self.predict_sample(x,tree["left"])
            else:
                return self.predict_sample(x,tree["right"])
        return tree
    
    def predict(self, X):
        return [self.predict_sample(x, self.tree) for x in X]


data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal'],
    'Windy': ['No', 'Yes', 'No', 'No', 'No'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes']
}
df = pd.DataFrame(data)

for col in df.columns:
    df[col] = pd.Categorical(df[col]).codes

X = df.drop('Play', axis=1).values
y = df['Play'].values

dt = DT(max_depth=3)
dt.fit(X, y)

predictions = dt.predict(X)
print("Predictions:", predictions)
print("Actual:", y)


    