import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Carga el dataset
data = pd.read_csv("dataset.csv", header=None)

# Ajusta los nombres según el número real de columnas
num_cols = data.shape[1]
data.columns = [f"v{i}" for i in range(num_cols - 1)] + ["label"]

# Divide entre características (X) y etiquetas (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Entrena el modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Guarda el modelo
with open("modelo_señas.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Modelo entrenado y guardado como 'modelo_señas.pkl'")
