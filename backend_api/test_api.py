import requests
import json

## adres lokalnego (włączonego) serwera
url = 'http://127.0.0.1:5000/chat'

## przykładowe zapytanie użytkownika
payload = {
    "query": "effectiveness of masks in children"
}

print(f"Wysyłam zapytanie do: {url}")
print(f"Treść: {payload}")

try:
    # wysłanie żądania POST
    response = requests.post(url, json=payload)
    
    # sprawdzenie statusu
    if response.status_code == 200:
        print("\nSerwer odpowiedział.")
        data = response.json()
        
        print("\n--- ODPOWIEDŹ AI ---")
        print(data['answer'])
        
        print("\n--- ŹRÓDŁA ---")
        for source in data['sources']:
            print(f"- {source['title']} (Score: {source['score']:.4f})")
    else:
        print(f"\nBłąd! Kod: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"\nBłąd połączenia: {e}")