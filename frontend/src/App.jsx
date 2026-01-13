import { useState } from "react";
import "./App.css";

function App() {
  // Stany aplikacji (zmienne, które zmieniają wygląd strony)
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Funkcja obsługująca wyszukiwanie
  const handleSearch = async () => {
    if (!query) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // 1. Wysyłanie zapytania do Flaska (backendu)
      const response = await fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: query }),
      });

      // 2. Obsługa błędów HTTP
      if (!response.ok) {
        throw new Error("Błąd połączenia z serwerem");
      }

      // 3. Odbieranie danych
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-wrapper">
      <div className="container">
        <header>
          <h1>Asystent COVID-19 (RAG)</h1>
          <p className="subtitle">
            Wyszukiwarka oparta na 850k+ artykułach naukowych
          </p>
        </header>

        <main>
          <div className="search-section">
            <p className="hint-text">
              Wpisz pytanie po polsku lub (dla lepszych wyników) po angielsku:
            </p>
            <div className="search-box">
              <input
                type="text"
                placeholder="np. effectiveness of masks in children..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSearch()}
                disabled={loading}
              />
              <button onClick={handleSearch} disabled={loading}>
                {loading ? "Analizuję..." : "Szukaj"}
              </button>
            </div>
          </div>

          {error && <div className="error-box">⚠️ {error}</div>}

          {result && (
            <div className="results-section">
              <div className="ai-answer">
                <h2>Odpowiedź AI:</h2>
                <p>{result.answer}</p>
              </div>

              <div className="sources-list">
                <h3>Źródła (Kontekst):</h3>
                <ul>
                  {result.sources.map((source, index) => (
                    <li key={index}>
                      <strong>
                        {index + 1}. {source.title}
                      </strong>
                      <span className="score">
                        {" "}
                        (Trafność: {(source.score * 100).toFixed(1)}%)
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
