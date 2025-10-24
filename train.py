import os
import urllib.parse
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from tempfile import TemporaryDirectory
from mlflow.models import infer_signature


def _normalize_file_uri(uri: str) -> str:
    """
    Normalize a file:// URI that may be Windows-style (file:///C:/...)
    into a WSL/Linux-friendly file:// URI (file:///mnt/c/...).

    If the URI is not a file:// URI, returns it unchanged.
    """
    if not uri:
        return uri
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "file":
        return uri

    # parsed.path typically like '/C:/Users/peper/...' or '/mnt/c/...'
    path = urllib.parse.unquote(parsed.path)

    # already WSL-style (starts with /mnt/)
    if path.startswith("/mnt/"):
        return "file://" + path

    # Windows-style absolute path exposed as /C:/...
    # Detect pattern like '/C:/' or 'C:/'
    stripped = path.lstrip("/")
    if len(stripped) >= 2 and stripped[1] == ":":
        drive = stripped[0].lower()
        rest = stripped[2:]  # the part after 'C:'
        # Ensure rest begins with '/'
        if not rest.startswith("/"):
            rest = "/" + rest
        new_path = f"/mnt/{drive}{rest}"
        return "file://" + new_path

    # otherwise, return original
    return uri


def _is_writable_file_uri(uri: str) -> bool:
    """
    Return True if the file:// URI points to a writable location from this process.
    """
    try:
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "file":
            return False
        path = pathlib.Path(urllib.parse.unquote(parsed.path))
        # check parent exists or is creatable
        parent = path if path.is_dir() else path.parent
        # If parent doesn't exist try to see if we can create the directory in cwd (avoid root)
        if parent.exists():
            return os.access(str(parent), os.W_OK)
        # Otherwise see if parent is inside cwd
        cwd = pathlib.Path.cwd().resolve()
        try_parent = parent.resolve()
        return str(try_parent).startswith(str(cwd))
    except Exception:
        return False


# ---------- Tracking URI setup (robust) ----------
cwd = pathlib.Path.cwd().resolve()
artifact_root_dir = cwd / "mlruns_wsl"  # safe local artifact root inside repo

env_tracking = os.environ.get("MLFLOW_TRACKING_URI")
tracking_uri = None

if env_tracking:
    # Normalize file URIs if necessary
    tracking_uri = _normalize_file_uri(env_tracking)
    # If it's a file:// that looks non-writable from this runner, fallback to local mlruns_wsl
    parsed = urllib.parse.urlparse(tracking_uri)
    if parsed.scheme == "file":
        if not _is_writable_file_uri(tracking_uri):
            # Can't write to that file URI from this runner -> fall back to local mlruns_wsl
            tracking_uri = f"file://{artifact_root_dir}"
else:
    # No env var: use a safe local mlruns directory inside the repo
    tracking_uri = f"file://{artifact_root_dir}"

# Set tracking URI
mlflow.set_tracking_uri(tracking_uri)

# Create/set experiment. If using a remote tracking server (http/https), let MLflow handle artifact root.
exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "wsl-experiment")
try:
    existing = mlflow.get_experiment_by_name(exp_name)
    if existing is None:
        # If our tracking URI is a local file URI, explicitly set artifact_location inside repo to avoid Windows paths
        parsed = urllib.parse.urlparse(tracking_uri)
        if parsed.scheme == "file":
            mlflow.create_experiment(
                exp_name, artifact_location=f"file://{artifact_root_dir}"
            )
        else:
            mlflow.create_experiment(exp_name)  # remote server will set artifact root
    mlflow.set_experiment(exp_name)
except Exception:
    # If experiment creation fails (race, permissions, etc.), continue — set_experiment may still work
    try:
        mlflow.set_experiment(exp_name)
    except Exception:
        pass

# ---------- Load data ----------
try:
    iris = pd.read_csv("data/iris_dataset.csv")
except FileNotFoundError:
    raise SystemExit("Error: El archivo 'data/iris_dataset.csv' no fue encontrado.")

X = iris.drop("target", axis=1)
y = iris["target"]

# ---------- Train and log ----------
with mlflow.start_run() as run:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save pickle locally
    joblib.dump(model, "model.pkl")

    # Save model in MLflow format to a temp dir and log as artifacts (avoids server endpoints some hosts don't support)
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_test.iloc[:5]

    try:
        with TemporaryDirectory() as tmpdir:
            mlflow.sklearn.save_model(
                model,
                tmpdir,
                input_example=input_example,
                signature=signature,
            )
            # Log artifacts under artifact_path "random-forest-model"
            mlflow.log_artifacts(tmpdir, artifact_path="random-forest-model")
    except Exception as e:
        # Log the error but continue (so metrics still get recorded)
        print("Warning: could not log MLflow model artifacts:", e)

    # Log params & metrics
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("accuracy", accuracy)

    # Optionally print run info
    try:
        run_id = run.info.run_id
        tracking_uri_used = mlflow.get_tracking_uri()
        print(f"MLflow run_id: {run_id}")
        print(f"MLflow tracking URI: {tracking_uri_used}")
    except Exception:
        pass

    print(f"Modelo entrenado y precisión: {accuracy:.4f}")
    print("Experimento registrado con MLflow.")

# ---------- Save metrics to a small text file for CML -->
with open("metrics.txt", "w", encoding="utf-8") as mf:
    mf.write(f"accuracy: {accuracy:.4f}\n")
    mf.write("n_estimators: 200\n")

# ---------- Create confusion matrix plot (artifact for CML) ----------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión")
plt.xlabel("Predicciones")
plt.ylabel("Valores Reales")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Matriz de confusión guardada como 'confusion_matrix.png'")

# Optionally also log the confusion matrix as an MLflow artifact (best-effort)
try:
    mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")
    mlflow.log_artifact("metrics.txt", artifact_path="metrics")
except Exception:
    # If tracking is remote and logging fails, ignore (the runner may not have access)
    pass
