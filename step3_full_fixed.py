# -*- coding: utf-8 -*-
import time
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
 
def train_and_log_model(n_estimators, max_depth, model_name):
    t0 = time.time()
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            model, name=model_name, serialization_format="pickle"
        )
        run = mlflow.active_run()
        print(f"[run] name={run.info.run_name} id={run.info.run_id}")
        print(f"Logged RandomForest model with n_estimators={n_estimators}, "
              f"max_depth={max_depth}, accuracy={accuracy:.4f}  ({time.time()-t0:.1f}s)")
        return run.info.run_id
 
print("Training and Logging Model 1...")
run_id_1 = train_and_log_model(20, 5, "random_forest_model_run_1")
 
print("\nTraining and Logging Model 2...")
run_id_2 = train_and_log_model(150, 10, "random_forest_model_run_V1_0_2")
 
# ── 注册模型到 Model Registry ──
model_uri = f"runs:/{run_id_2}/random_forest_model_run_V1_0_2"
mv = mlflow.register_model(model_uri=model_uri, name="RandomForestDigits")
print("[register] name=%s version=%s status=%s" % (mv.name, mv.version, mv.status))
 
# ── 从注册表加载并预测 ──
loaded = mlflow.pyfunc.load_model("models:/RandomForestDigits/1")
print("[predict] registry(pyfunc) =", list(loaded.predict(X_test[:5])))
print("[predict] 真实标签           =", list(y_test[:5]))