import numpy as np

class LogisticRegressor():
   def __init__(self, learning_rate=0.1, max_iter=1000, tolerance=1e-3):
      self.learning_rate = learning_rate
      self.tolerance = tolerance
      self.max_iter = max_iter
      self.weights = None
      self.bias = 0
      self.errors_history = []  # Track loss during training
      
   def sigmoid(self, z):
      """
      Compute the sigmoid function to transform linear output into probabilities.
      """
      z = np.clip(z, -500, 500) # Prevent overflow
      return 1 / (1 + np.exp(-z))

   def fit(self, X, y):
      """
      Fit the model using vectorized Stochastic Gradient Descent.
      """
      n_samples, n_features = X.shape
      self.weights = np.zeros(n_features)

      for iteration in range(self.max_iter):
         # Compute predictions for all samples
         y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)

         y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
         # Compute errors for all samples
         errors = y_pred - y

         # Compute gradients (vectorized) with stability check
         gradient_w = np.dot(X.T, errors) / (n_samples + 1e-8)
         gradient_b = np.sum(errors) / (n_samples + 1e-8)

         # Update parameters
         self.weights -= self.learning_rate * gradient_w
         self.bias -= self.learning_rate * gradient_b

         # Record Binary Cross Entropy for the learning curve
         loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
         self.errors_history.append(loss)

         # Debugging: Log progress and gradient norms
         if iteration % 10 == 0:
            gradient_norm = np.linalg.norm(gradient_w)
            print(f"Iteration {iteration}, Loss = {loss}, Gradient_w Norm = {gradient_norm}, Gradient_b = {gradient_b}")

         if np.linalg.norm(gradient_w) < self.tolerance:
            print(f"Converged at iteration {iteration}")
            break

   def predict(self, X):
      """
      Predict target values using the learned weights and bias.
      """
      y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
      return (y_pred >= 0.5).astype(int)