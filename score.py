import json
import joblib
import numpy as np
import os
from azureml.core.model import Model

### wywoływane, gdy punkt końcowy jest ładowany
def init():
    global model
    ##### składanie ścieżki do modelu, który Azure automatycznie pobrał
    ##### nazwa modelu musi być taka sama, jak ta zarejestrowana w Azure
    model_path = Model.get_model_path(model_name='Covid-Topic-Classifier-LGBM-gbdt')
    
    ### ładownie modelu
    model = joblib.load(model_path)
    print("Model załadowany.")

### wywoływane, przy każdym zapytaniu do API
def run(raw_data):
    try:
        ##### przetwarzanie danych wejściowych (spodziewany format JSON)
        data = json.loads(raw_data)['data']
        ##### konwersja listy na tablicę NumPy której oczekuje model
        data_np = np.array(data)
        
        ##### uruchomienie predykcji
        predictions = model.predict(data_np)
        
        ##### zwrócenie wyników w formacie JSON
        return json.dumps({"predictions": predictions.tolist()})
        
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})