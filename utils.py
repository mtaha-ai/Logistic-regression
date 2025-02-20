from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Metrics:
    def compute_confusion_matrix(self, y_true, y_pred):
        """
        Compute confusion matrix values: TP, TN, FP, FN.
        """
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        return TP, TN, FP, FN
    
    def recall(self, TP=None, FN=None):
        """
        Compute recall metric.
        """
        return TP / (TP+FN)
    
    def precision(self, TP=None, FP=None):
        """
        Compute precision metric.
        """
        return TP / (TP+FP)
    
    def f1_score(self, y_true, y_pred):
        """
        Compute F1 score from precision and recall.
        """
        TP, TN, FP, FN = self.compute_confusion_matrix(y_true, y_pred)
        precision = self.precision(TP=TP, FP=FP)
        recall = self.recall(TP=TP, FN=FN)
        return (2*precision*recall) / (precision+recall)

    
class Scaler:
    def MinMaxScaler(self, data):
        """
        Scale the input data to a range between 0 and 1 using Min-Max Scaling.
        :param data: Input data, shape (n_samples,)
        :return: Scaled data, shape (n_samples,)
        """
        data_min = data.min()
        data_max = data.max()
        data_scaled = (data - data_min) / (data_max - data_min)
        return data_scaled

class Plot:
    @staticmethod
    def plot_learning_curve(errors_history):
        """
        Plot the learning curve using the recorded errors history.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(errors_history, label='Training Error Binary Cross Entropy', linewidth=2)
        plt.title('Learning Curve', fontsize=16)
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Binary Cross Entropy', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.show()
    
    @staticmethod
    def plot_scatter(X, y, xlabel, ylabel, title):
        """
        Plot the scatter plot using data points.
        """
        sc = plt.scatter(X[:, 0], X[:, 1], cmap=ListedColormap(["red", "green"]), c=y, alpha=0.5)
        cbar = plt.colorbar(sc, ticks=[0, 1])
        cbar.ax.set_yticklabels(["No Purchase", "Purchase"])

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)

        # Show plot
        plt.show()
    
    @staticmethod
    def plot_decision_boundary(X, y, w, b):
        X1 = X[:, 0]
        X2 = X[:, 1]
        colors = ['green' if label == 1 else 'red' for label in y]
        # Scatter plot of actual data points
        plt.figure(figsize=(7, 5))
        plt.scatter(X1, X2, c=colors, alpha=0.6, label="Data Points")
        
        # Generate X1 values for decision boundary
        x1_vals = np.linspace(X1.min(), X1.max(), 100)
        # Compute corresponding X2 values (Age) for decision boundary
        x2_vals = -(b + w[0] * x1_vals) / w[1]
        # Plot decision boundary
        plt.plot(x1_vals, x2_vals, color="black", linestyle="--", label="Decision Boundary")
        # Labels and title
        plt.xlabel("Ad Watch Time (seconds)")
        plt.ylabel("Age")
        plt.title("Decision Boundary for Logistic Regression")
        plt.legend()
        plt.grid(True)
        # Show plot
        plt.show()

    @staticmethod
    def plot_confusion_matrix(TP, TN, FP, FN):
        confusion_matrix = np.array([[TP, FP], 
                                 [FN, TN]])
        labels = ["Positive", "Negative"]
        plt.figure(figsize=(5, 4))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.title("Confusion Matrix")
        plt.show()