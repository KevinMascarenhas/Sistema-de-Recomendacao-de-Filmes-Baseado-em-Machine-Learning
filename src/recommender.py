from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import re
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path("data/movies.csv")
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "movie_recommender.pkl"


def _normalize_text(text: str) -> str:
    # Limpa pontuacao e padroniza o texto para melhorar a comparacao entre filmes.
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _join_tokens(row: pd.Series) -> str:
    # Junta os atributos mais descritivos do filme em uma unica representacao textual.
    parts = [
        str(row["genre"]),
        str(row["overview"]),
        str(row["cast"]),
        str(row["director"]),
    ]
    return _normalize_text(" ".join(parts))


@dataclass
class MovieRecommender:
    movies: pd.DataFrame
    similarity_matrix: object

    @classmethod
    def train_from_csv(cls, csv_path: str | Path = DATA_PATH) -> "MovieRecommender":
        movies = pd.read_csv(csv_path)
        required_columns = {"title", "genre", "overview", "cast", "director", "year", "rating"}
        missing = required_columns.difference(movies.columns)
        if missing:
            raise ValueError(f"Dataset incompleto. Faltam as colunas: {sorted(missing)}")

        movies = movies.copy()
        # Cria a coluna que servira de entrada para a vetorizacao e o calculo de similaridade.
        movies["tags"] = movies.apply(_join_tokens, axis=1)

        # Transforma texto em vetores numericos e mede o quanto cada filme se parece com os demais.
        vectorizer = CountVectorizer(max_features=5000, stop_words="english")
        vectors = vectorizer.fit_transform(movies["tags"]).toarray()
        similarity = cosine_similarity(vectors)

        return cls(movies=movies, similarity_matrix=similarity)

    def save(self, model_path: str | Path = MODEL_PATH) -> Path:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as file:
            # Persiste os dados essenciais para evitar novo treino a cada execucao.
            pickle.dump(
                {
                    "movies": self.movies.drop(columns=["tags"], errors="ignore"),
                    "similarity_matrix": self.similarity_matrix,
                },
                file,
            )
        return model_path

    @classmethod
    def load(cls, model_path: str | Path = MODEL_PATH) -> "MovieRecommender":
        with Path(model_path).open("rb") as file:
            payload = pickle.load(file)
        movies = payload["movies"].copy()
        # Recria as tags para manter o mesmo formato usado durante o treino.
        movies["tags"] = movies.apply(_join_tokens, axis=1)
        return cls(movies=movies, similarity_matrix=payload["similarity_matrix"])

    def get_movie(self, title: str) -> dict[str, Any]:
        if title not in set(self.movies["title"]):
            raise ValueError(f'Filme "{title}" nao encontrado no dataset.')

        movie = self.movies.loc[self.movies["title"] == title].iloc[0]
        return {
            "title": movie["title"],
            "genre": movie["genre"],
            "year": int(movie["year"]),
            "rating": float(movie["rating"]),
            "overview": movie["overview"],
            "cast": movie["cast"],
            "director": movie["director"],
        }

    def recommend(self, title: str, top_n: int = 5) -> list[dict[str, Any]]:
        if title not in set(self.movies["title"]):
            raise ValueError(f'Filme "{title}" nao encontrado no dataset.')

        movie_index = self.movies.index[self.movies["title"] == title][0]
        distances = list(enumerate(self.similarity_matrix[movie_index]))
        ranked = sorted(distances, key=lambda item: item[1], reverse=True)

        # Ignora o proprio filme e devolve apenas os vizinhos mais proximos.
        recommendations: list[dict[str, Any]] = []
        for index, score in ranked[1 : top_n + 1]:
            movie = self.movies.iloc[index]
            recommendations.append(
                {
                    "title": movie["title"],
                    "genre": movie["genre"],
                    "year": int(movie["year"]),
                    "rating": float(movie["rating"]),
                    "score": round(float(score), 3),
                    "overview": movie["overview"],
                    "cast": movie["cast"],
                    "director": movie["director"],
                }
            )
        return recommendations
