import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("rf.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
y = y.map(lambda x: 1 if x == 'Canceled' else 0)

model = RandomForestClassifier()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
