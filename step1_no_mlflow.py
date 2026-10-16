from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
 
data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)
 
def train_model(n_estimators, max_depth):
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model with n_estimators={n_estimators}, max_depth={max_depth} "
          f"achieved accuracy={accuracy:.4f}")
    return model, accuracy
 
print("Training Model 1...")
model_1, acc_1 = train_model(n_estimators=20, max_depth=5)
 
print("\nTraining Model 2...")
model_2, acc_2 = train_model(n_estimators=100, max_depth=10)