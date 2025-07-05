import React, { useState } from 'react';
import axios from 'axios';
import { Loader2 } from 'lucide-react';
import './App.css';

const API_BASE = process.env.REACT_APP_API || 'http://localhost:8000';

function App() {
  const [text, setText] = useState('');
  const [emotions, setEmotions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function analyse() {
    setLoading(true);
    setError('');
    setEmotions([]);
    try {
      const { data } = await axios.post(`${API_BASE}/predict`, { text });
      setEmotions(data.emotions);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        'Unexpected server error'
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-container">
      <h1>Sentiment-Checker</h1>

      <textarea
        placeholder="Type or paste text here…"
        value={text}
        onChange={e => setText(e.target.value)}
      />

      <button
        onClick={analyse}
        disabled={loading || !text.trim()}
      >
        {loading
          ? <><Loader2 className="spinner" /> Analysing…</>
          : 'Analyse'}
      </button>

      {error && <div className="error">Error: {error}</div>}

      {emotions.length > 0 && (
        <div className="results">
          <h2>Detected Emotions</h2>
          <ul>
            {emotions.map(e => <li key={e}>{e}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
