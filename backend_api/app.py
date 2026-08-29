import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pinecone import Pinecone
from groq import Groq
###from sentence_transformers import SentenceTransformer ### niepotrzebne bo przeliczanie na wektory promptu użytkownika poprzez HuggingFace
###import numpy as np ### niepotrzebne bo przeliczanie na wektory promptu użytkownika poprzez HuggingFace
from huggingface_hub import InferenceClient


## konfiguracja
load_dotenv() ### klucze z pliku .env


app = Flask(__name__)
CORS(app) ### pozwala na zapytania z innej domeny (Reacta)


### konfiguracja zewnętrznych serwisów

print("Inicjalizacja Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("main-projects-index") ### nazwa indeksu
NAMESPACE = "covid-papers" ### nazwa namespace

print("Inicjalizacja Groq...")
client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))


print("Inicjalizacja Hugging Face Client...")
### Hugging Face Client sam zadba o poprawny adres URL (router vs api-inference)
client_hf = InferenceClient(token=os.getenv("HF_API_KEY"))


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

        ## generowanie wektora przez Hugging Face API (zamiast tworzenie lokalnie) --> to oszczędza RAM

        try:
            ## feature_extraction automatycznie robi zapytanie do modelu
            ## model="sentence-transformers/allenai-specter"
            embedding_response = client_hf.feature_extraction(
                user_query,
                model="sentence-transformers/allenai-specter"
            )

            ## biblioteka zwraca ndarray (numpy), trzeba zamienić na listę
            ## wynik może być zagnieżdżony [[0.1, 0.2...]], bierzemy pierwszy wektor
            if embedding_response.ndim > 1:
                query_embedding = embedding_response[0].tolist()
            else:
                query_embedding = embedding_response.tolist()

        except Exception as e:
            print(f"Błąd generowania embeddingu (HF): {e}")
            return jsonify({"error": f"Błąd modelu AI: {str(e)}"}), 500

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
            model="qwen/qwen3.6-27b", ##### darmowy i szybki model dostępny na Groq
        )

        ai_response = chat_completion.choices[0].message.content

        ## zwrócenie wyników do Frontendu
        return jsonify({
            "answer": ai_response,
            "sources": sources
        })

    except Exception as e:
        print(f"Błąd ogólny: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    #### uruchomienie serwera lokalnie na porcie 5000
    app.run(debug=True, port=5000)