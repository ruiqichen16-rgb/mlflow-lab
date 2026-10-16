import sys
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
 
URI = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5010"
mlflow.set_tracking_uri(URI)
print("[tracking_uri]", mlflow.get_tracking_uri())
 
data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
 
mlflow.set_experiment("RemoteDemo")
with mlflow.start_run(run_name="remote_run_1"):
    model = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("max_depth", 8)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, name="remote_rf",
                             serialization_format="pickle")
    print("[accuracy] %.4f" % acc)