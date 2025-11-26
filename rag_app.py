import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import joblib
import torch
from openai import OpenAI

# konfiguracja aplikacji
st.set_page_config(page_title="Asystent Badawczy AI (RAG)", layout="wide")
st.title("Asystent Badawczy AI: Artykuły COVID-19")
st.markdown("Zadaj pytanie, a sztuczna inteligencja odpowie na podstawie ponad 850 tys. artykułów naukowych, wykorzystując metodę RAG (Retrieval-Augmented Generation).")

# pasek boczny - ustawienia
st.sidebar.header("Ustawienia")
mode = st.sidebar.radio("Tryb działania:", ["Proste Wyszukiwanie (Tytuły)", "Czat z AI (RAG)"])
n_context = st.sidebar.slider("Liczba artykułów do kontekstu:", 3, 20, 5)

# dodatkowy suwak temperatury (kreatywności)
temperature = st.sidebar.slider("Kreatywność (Temperatura):", min_value=0.0, max_value=1.0, value=0.7, step=0.1, help="Niższa = bardziej precyzyjna/deterministyczna. Wyższa = bardziej kreatywna.")

# ładowanie zasobów (Cache)
@st.cache_resource
def load_resources():
    print("Ładowanie modelu SentenceTransformer...")
    # użyj 'cuda' jeśli dostępne, w przeciwnym razie 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('allenai-specter', device=device)
    
    print("Ładowanie DataFrame z tytułami/abstraktami...")
    df = joblib.load('df_with_all_clusters.joblib')
    
    print("Łączenie z bazą ChromaDB...")
    # połączenie z istniejącą bazą ChromaDB
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name="covid_articles")
    
    print("Zasoby załadowane.")
    return model, df, collection

# inicjalizacja zasobów
try:
    model, df, collection = load_resources()
except Exception as e:
    st.error(f"Błąd ładowania zasobów: {e}")
    st.stop()

# funkcje pomocnicze

def get_relevant_docs(query_text, n_results=5):
    """
    Pobiera identyfikatory i odległości relewantnych dokumentów z ChromaDB.
    """
    if not query_text:
        return [], []

    # konwersja zapytania na embedding
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    query_embedding = model.encode(query_text, convert_to_tensor=True, device=device)
    query_embeddings_list = [query_embedding.cpu().numpy().tolist()]

    # zapytanie do ChromaDB
    results = collection.query(
        query_embeddings=query_embeddings_list,
        n_results=n_results,
        include=['distances'] # potrzeba ID i odległości
    )

    if results and results.get('ids') and results['ids'][0]:
        found_ids = [int(id_str) for id_str in results['ids'][0]]
        distances = results['distances'][0]
        return found_ids, distances
    else:
        return [], []

def query_llm(prompt, temp=0.7):
    """
    Wysyła prompt do lokalnego LLM poprzez API LM Studio.
    """
    # inicjalizacja klienta OpenAI wskazującego na lokalny serwer LM Studio
    client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
    
    try:
        completion = client.chat.completions.create(
            model="model-identifier", # ignorowane przez LM Studio, używa załadowanego modelu
            messages=[
                {"role": "system", "content": "Jesteś pomocnym i precyzyjnym asystentem badawczym specjalizującym się w literaturze na temat COVID-19. Zawsze odpowiadaj w języku polskim na podstawie dostarczonego kontekstu, chyba że użytkownik poprosi wyraźnie o odpowiedź w innym konkretnym języku (np. angielskim). Jeśli odpowiedzi nie ma w kontekście, poinformuj o tym."},
                {"role": "user", "content": prompt}
            ],
            temperature=temp, # użycie temperatury z ustawień suwaka
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Błąd komunikacji z LLM: {e}. Upewnij się, że serwer LM Studio działa na porcie 1234."

#### Główna Logika Aplikacji 

query = st.text_input("Wpisz pytanie lub słowa kluczowe (po polsku albo najlepiej po angielsku):", 
                      placeholder="np. skuteczność maseczek u dzieci (albo najlepiej w wariancie angielskim: e.g., effectiveness of masks in children)")

if st.button("Szukaj / Zapytaj"):
    if not query:
        st.warning("Proszę wpisać zapytanie.")
    else:
        with st.spinner("Przeszukuję bazę wiedzy..."):
            # 1. Retrieval (Wyszukiwanie)
            doc_ids, distances = get_relevant_docs(query, n_results=n_context)
            
            if not doc_ids:
                st.warning("Nie znaleziono pasujących artykułów.")
            else:
                # Pobranie szczegółów z DataFrame
                try:
                    # próba pobrania tytułów i abstraktów
                    articles_data = df.loc[doc_ids][['title', 'abstract']]
                except KeyError:
                    articles_data = df.loc[doc_ids][['title']]
                    articles_data['abstract'] = "Brak dostępnego abstraktu."

                # 2. Wyświetlanie wyników (Tryb Prosty)
                if mode == "Proste Wyszukiwanie (Tytuły)":
                    st.success(f"Znaleziono {len(doc_ids)} artykułów:")
                    
                    # przygotowanie tabeli do wyświetlenia
                    display_df = articles_data.copy()
                    display_df['Wskaźnik Trafności'] = distances # mniejsza odległość => wyższa trafność
                    st.dataframe(display_df, use_container_width=True)

                # 3. Generowanie Odpowiedzi (Tryb RAG)
                elif mode == "Czat z AI (RAG)":
                    st.info(f"Znaleziono {len(doc_ids)} artykułów kontekstowych. Generowanie odpowiedzi...")
                    
                    # budowanie kontekstu z pobranych artykułów
                    context_text = ""
                    for i, (index, row) in enumerate(articles_data.iterrows()):
                        context_text += f"Źródło {i+1}:\nTytuł: {row['title']}\nAbstrakt: {row['abstract']}\n\n"
                    
                    # konstrukcja Promptu dla LLM
                    full_prompt = f"""
                    Poniżej znajdują się informacje kontekstowe (fragmenty artykułów naukowych).
                    ---------------------
                    {context_text}
                    ---------------------
                    Na podstawie powyższego kontekstu i bez używania wcześniejszej wiedzy ogólnej, odpowiedz na pytanie użytkownika.
                    Odpowiedź musi być w języku polskim, chyba że użytkownik wyraźnie zaznaczy, że ma być w innym konkretnym języku (np. angielskim).

                    WAŻNE: Używaj wyłącznie alfabetu łacińskiego. Nie używaj cyrylicy ani znaków azjatyckich.
                    
                    Pytanie: {query}
                    Odpowiedź:
                    """
                    
                    # Generowanie
                    with st.spinner("AI analizuje i pisze odpowiedź..."):
                        answer = query_llm(full_prompt, temp=temperature) ### przekazuje temperaturę z suwaka
                    
                    # Wyświetlenie Odpowiedzi
                    st.markdown("Odpowiedź AI:")
                    st.write(answer)
                    
                    # Wyświetlenie Źródeł (Rozwijana lista)
                    with st.expander("Zobacz Źródła (Kontekst użyty do odpowiedzi)"):
                        for i, (index, row) in enumerate(articles_data.iterrows()):
                            st.markdown(f"**{i+1}. {row['title']}**")
                            # skracanie abstraktu jeśli jest bardzo długi
                            abstract_text = str(row['abstract'])
                            st.caption(abstract_text[:1000] + "..." if len(abstract_text) > 1000 else abstract_text)
                            st.divider()

st.markdown("---")
st.caption("Zasilane przez: AllenAI-Specter (Embeddingi), ChromaDB (Baza Wektorowa), Qwen 2.5 (Lokalny LLM via LM Studio)")