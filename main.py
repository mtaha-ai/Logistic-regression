from regressor import LogisticRegressor
from utils import Plot, Scaler, Metrics
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
model = LogisticRegressor()
scaler = Scaler()
metrics = Metrics()

X = df[["Ad_Watch_Time", "Age"]].values
y = df['Purchase'].values

Plot.plot_scatter(X, y, 'Ad Watch Time', 'Age', 'Purchases based on Ad Watch Time')

X_scaled = np.zeros_like(X)
X_scaled[:,0] = scaler.MinMaxScaler(X[:,0])
X_scaled[:,1] = scaler.MinMaxScaler(X[:,1])
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)

Plot.plot_learning_curve(model.errors_history)
Plot.plot_confusion_matrix(*metrics.compute_confusion_matrix(y, y_pred))
Plot.plot_decision_boundary(X_scaled, y, model.weights, model.bias)