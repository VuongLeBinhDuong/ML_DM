import joblib

def load_model(path, device=None):
    model = joblib.load(path)
    return model, None 