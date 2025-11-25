import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import joblib
import torch

## konfiguracja
st.set_page_config(page_title="Wyszukiwarka Semantyczna COVID-19", layout="wide")
st.title("Wyszukiwarka Semantyczna Artykułów COVID-19")
st.markdown("Wpisz zapytanie w języku naturalnym, aby znaleźć najbardziej pasujące tematycznie artykuły naukowe.")

## Ładowanie zasobów (model, baza, DataFrame)
## @st.cache_resource --> żeby załadować ciężkie obiekty tylko raz

@st.cache_resource
def load_resources():
    print("Ładowanie modelu SentenceTransformer...")
    model = SentenceTransformer('allenai-specter', device='cuda')
    print("Ładowanie DataFrame z tytułami...")
    df = joblib.load('df_with_all_clusters.joblib')
    print("Łączenie z bazą ChromaDB...")
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name="covid_articles")
    print("Zasoby załadowane.")
    return model, df, collection

model, df, collection = load_resources()

## Funkcja wyszukująca
def search_similar_articles_streamlit(query_text, n_results=10):
    """
    Wyszukuje w bazie ChromaDB i zwraca wyniki jako DataFrame.
    """
    if not query_text:
        return pd.DataFrame() ### pusty DataFrame jeśli zapytanie jest puste

    ### konwersja zapytania na embedding (GPU)
    query_embedding = model.encode(query_text, convert_to_tensor=True, device='cuda')
    query_embeddings_list = [query_embedding.cpu().numpy().tolist()] ### przeniesienie na CPU do wyszukiwania

    ### zapytanie
    results = collection.query(
        query_embeddings=query_embeddings_list,
        n_results=n_results,
        include=['distances']
    )

    ### wyniki
    if results and results.get('ids') and results['ids'][0]:
        found_ids = [int(id_str) for id_str in results['ids'][0]]
        distances = results['distances'][0] if results.get('distances') else [None] * len(found_ids)

        ### tworzenie DataFrame
        found_titles = df.loc[found_ids][['title']] #### tylko kolumna 'title'
        results_df = found_titles.copy()
        results_df['ID'] = found_ids
        results_df['Odległość'] = distances
        results_df = results_df[['ID', 'Odległość', 'title']] #### zmiana kolejności kolumn
        results_df.rename(columns={'title': 'Tytuł'}, inplace=True)
        return results_df
    else:
        return pd.DataFrame() #### pusty DataFrame jeśli brak wyników

## Interfejs

### pole do wpisania zapytania
query = st.text_input("Wpisz swoje zapytanie:", placeholder="np. skuteczność maseczek u dzieci")

### przycisk do uruchomienia wyszukiwania
if st.button("Szukaj podobnych artykułów"):
    with st.spinner("Przeszukuję bazę danych..."): #### pokazuje animację ładowania
        search_results_df = search_similar_articles_streamlit(query, n_results=15)

    if not search_results_df.empty:
        st.success(f"Znaleziono {len(search_results_df)} pasujących artykułów:")
        ##### wyniki jako interaktywna tabela
        st.dataframe(search_results_df, use_container_width=True)
    elif query: ###### jeśli było zapytanie ale nic nie znaleziono
        st.warning("Nie znaleziono żadnych wyników dla podanego zapytania.")
    ###### jeśli query było puste nic nie wyświetla (bo search_similar_articles_streamlit zwróciło pusty df) ######

st.markdown("---")
st.caption("Aplikacja wykorzystuje model 'allenai-specter' i bazę wektorową ChromaDB.")
