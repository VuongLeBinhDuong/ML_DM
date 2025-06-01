import joblib
import xgboost as xgb
import os

def load_model(path, device=None):
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    
    try:
        # Try loading with joblib first
        model = joblib.load(path)
        
        # If it's an XGBoost model, save it in the new format
        if isinstance(model, (xgb.Booster, xgb.XGBClassifier)):
            new_path = path.replace('.pkl', '_new.json')
            if isinstance(model, xgb.XGBClassifier):
                # Get the underlying booster
                booster = model.get_booster()
                booster.save_model(new_path)
            else:
                model.save_model(new_path)
            print(f"Model has been saved in new format at {new_path}")
        return model, None
    except Exception as e:
        print(f"Error loading model with joblib: {e}")
        try:
            # Try loading directly with XGBoost
            model = xgb.Booster()
            model.load_model(path)
            return model, None
        except Exception as e:
            print(f"Error loading model with XGBoost: {e}")
            raise 