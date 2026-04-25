import joblib

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def log_message(msg):
    print(f"[INFO] {msg}")