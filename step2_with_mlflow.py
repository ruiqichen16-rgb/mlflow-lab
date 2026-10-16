import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
 
print("[mlflow]", mlflow.__version__)
print("[tracking_uri]", mlflow.get_tracking_uri())
 
data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
 
experiment_name = "RandomForest_V1.0.1"
mlflow.set_experiment(experiment_name)
 
def train_and_log_model(n_estimators, max_depth, run_name):
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
 
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, run_name)      # ← 讲义原稿写法，本机必失败
 
        print(f"Logged RandomForest model with n_estimators={n_estimators}, "
              f"max_depth={max_depth}, accuracy={accuracy:.4f}")
 
print("Training and Logging Model 1...")
train_and_log_model(n_estimators=20, max_depth=5, run_name="random_forest_model_run_1")
 
print("\nTraining and Logging Model 2...")
train_and_log_model(n_estimators=100, max_depth=10, run_name="random_forest_model_run_V1.0.2")