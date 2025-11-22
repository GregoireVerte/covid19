import json
import joblib
import numpy as np
import os

# zmienna ta zostanie automatycznie ustawiona przez Azure
MODEL_DIR = os.getenv('AZUREML_MODEL_DIR')

# funkcja ta jest wywoływana raz gdy kontener startuje
def init():
    global model
    
    # nazwa pliku dokładnie taka jaką wgraliśmy
    model_filename = 'lgb_gbdt_best_model_ver4.joblib'
    
    # pełna ścieżka
    model_path = os.path.join(MODEL_DIR, model_filename)
    
    # sprawdzenie czy plik istnieje
    if not os.path.exists(model_path):
        # jeśli go nie ma wypisuje zawartość folderu aby pomóc w debugowaniu
        print(f"BŁĄD: Nie znaleziono pliku {model_filename} w ścieżce {MODEL_DIR}")
        print("Zawartość folderu modelu:")
        print(os.listdir(MODEL_DIR))
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # ładowanie modelu
    model = joblib.load(model_path)
    print(f"Model załadowany pomyślnie z: {model_path}")

# ta funkcja jest wywoływana przy każdym zapytaniu
def run(raw_data):
    global model
    if model is None:
        return json.dumps({"error": "Model not loaded"})
        
    try:
        data = json.loads(raw_data)['data']
        input_data = np.array(data)
        
        predictions = model.predict(input_data)
        
        return json.dumps({"predictions": predictions.tolist()})
        
    except Exception as e:
        return json.dumps({"error": str(e)})