import pickle
import os
import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)  #Ensure folder exists
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise e
        
    logging.info(f"Preprocessor object saved at: {file_path}")
    print(f"Pickle saved at {file_path}")

