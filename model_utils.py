import joblib

def save_model(model, filename='graduation_predictor_model.pkl'):
    """Save the trained model to a file."""
    joblib.dump(model, filename)
    print(f"Model saved successfully to {filename}")

def load_model(filename='graduation_predictor_model.pkl'):
    """Load a trained model from a file."""
    model = joblib.load(filename)
    print(f"Model loaded successfully from {filename}")
    return model
