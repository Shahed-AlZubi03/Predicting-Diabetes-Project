import mlflow.sklearn
from sklearn.metrics import accuracy_score
import pandas as pd

# Replace with your actual run_id
run_id = "7203a09f6bc6487791de609807f03433"

model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# Reload your data
df = pd.read_csv(r"C:\Users\Shahe\OneDrive\Desktop\ml_project\data\diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

y_pred = model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))
