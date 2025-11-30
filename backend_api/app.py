import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pinecone import Pinecone
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np

## konfiguracja
load_dotenv() ### klucze z pliku .env

app = Flask(__name__)
CORS(app) ### pozwala na zapytania z innej domeny (Reacta)

## inicjalizacja zasobów (raz przy starcie serwera)

print("Inicjalizacja modelu embeddingów...")
### na serwerze (Render) nie ma GPU więc wymusza się CPU
model = SentenceTransformer('allenai-specter', device='cpu')

print("Inicjalizacja Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("main-projects-index") ### nazwa indeksu
NAMESPACE = "covid-papers" ### nazwa namespace

print("Inicjalizacja Groq...")
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

## ENDPOINTY API

@app.route('/')
def home():
    return "Serwer działa! API jest gotowe."

@app.route('/chat', methods=['POST'])
def chat():
    try:
        ## odbieranie danych od Frontendu (React)
        data = request.json
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({"error": "Brak zapytania"}), 400

        print(f"Otrzymano zapytanie: {user_query}")

        ## tworzenie Embeddingu zapytania
        ##### konwersja do listy bo Pinecone jej oczekuje (nie numpy array)
        query_embedding = model.encode(user_query).tolist()

        ## szukanie w Pinecone (Retrieval)
        search_results = index.query(
            namespace=NAMESPACE,
            vector=query_embedding,
            top_k=20, #### pobieranie 20 najlepszych artykułów
            include_metadata=True
        )

        ## budowanie kontekstu dla AI
        context_text = ""
        sources = []
        
        for match in search_results['matches']:
            meta = match['metadata']
            title = meta.get('title', 'Bez tytułu')
            abstract = meta.get('abstract', '')
            
            ## dodawanie do tekstu dla AI
            context_text += f"Tytuł: {title}\nAbstrakt: {abstract}\n\n"
            
            ## zapis źródła żeby wyświetlić użytkownikowi
            sources.append({"title": title, "score": match['score']})

        ## wysyłanie do Groq (Generation)
        system_prompt = """
        Jesteś asystentem naukowym specjalizującym się w COVID-19. 
        Odpowiadaj na pytanie użytkownika WYŁĄCZNIE na podstawie poniższego kontekstu i bez używania wcześniejszej wiedzy ogólnej.
        Jeśli w kontekście nie ma odpowiedzi, napisz "Nie znalazłem informacji w dostępnych artykułach".
        Odpowiadaj w języku polskim chyba, że użytkownik wyraźnie poprosi o odpowiedź w innym języku (np. angielskim).
        WAŻNE: Używaj wyłącznie alfabetu łacińskiego. Nie używaj cyrylicy ani znaków azjatyckich.
        """

        user_message = f"""
        Kontekst (artykuły naukowe):
        {context_text}

        Pytanie użytkownika: {user_query}
        """

        chat_completion = client_groq.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            model="llama3-8b-8192", ##### darmowy i szybki model dostępny na Groq
        )

        ai_response = chat_completion.choices[0].message.content

        ## zwrócenie wyników do Frontendu
        return jsonify({
            "answer": ai_response,
            "sources": sources
        })

    except Exception as e:
        print(f"Błąd: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    #### uruchomienie serwera lokalnie na porcie 5000
    app.run(debug=True, port=5000)