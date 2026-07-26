from __future__ import annotations

# Estrutura de dados usada para representar o modelo treinado
from dataclasses import dataclass
from pathlib import Path
import pickle
import re
from typing import Any

# Dependencias principais de processamento e similaridade
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cliente da API da TMDB para buscar dados de filmes externos
from integrations.tmdb_client import TMDBClient


# Localizacao do dataset CSV inicial
DATA_PATH = Path("data/movies.csv")
# Pasta onde o modelo serializado sera salvo
ARTIFACTS_DIR = Path("artifacts")
# Caminho do arquivo pickle do modelo treinado
MODEL_PATH = ARTIFACTS_DIR / "movie_recommender.pkl"


def _normalize_text(text: str) -> str:
    # Normaliza o texto do filme para que comparações sejam mais robustas.
    # 1. coloca tudo em minúsculo
    # 2. remove pontuação
    # 3. remove espaços extras
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _join_tokens(row: pd.Series) -> str:
    # Cria uma string textual única com os atributos mais relevantes do filme.
    # Essa string será convertida em vetores pelo CountVectorizer.
    parts = [
        str(row["genre"]),
        str(row["overview"]),
        str(row["cast"]),
        str(row["director"]),
    ]
    return _normalize_text(" ".join(parts))


DEFAULT_FIELD_WEIGHTS: dict[str, float] = {
    "genre": 3.0,
    "overview": 1.0,
    "cast": 2.0,
    "director": 2.0,
}


def _build_weighted_feature_matrix(
    movies: pd.DataFrame,
    field_weights: dict[str, float],
) -> object:
    # Converte cada campo em uma matriz de características separada.
    # Depois aplica o peso desejado em cada matriz e concatena horizontalmente.
    feature_matrices = []
    for field in ["genre", "overview", "cast", "director"]:
        raw_text = movies[field].fillna("").astype(str)
        vectorizer = CountVectorizer(max_features=5000, stop_words="english")
        matrix = vectorizer.fit_transform(raw_text)
        weight = field_weights.get(field, 1.0)
        if weight != 1.0:
            matrix = matrix.multiply(weight)
        feature_matrices.append(matrix)

    return hstack(feature_matrices)


def _build_tmdb_movie_record(movie_details: dict[str, Any]) -> dict[str, Any]:
    # Recebe os dados do TMDB e padroniza as informações do filme.
    # Isso evita que o restante da classe precise lidar com estruturas diferentes.
    credits = movie_details.get("credits", {})

    # Toma apenas os 5 primeiros atores do elenco para manter o registro simples.
    cast_names = [person["name"] for person in credits.get("cast", [])[:5]]

    # Localiza os nomes dos diretores no campo crew da API.
    directors = [
        person["name"]
        for person in credits.get("crew", [])
        if person.get("job") == "Director"
    ]

    # Coleta os nomes dos gêneros do filme.
    genres = [genre["name"] for genre in movie_details.get("genres", [])]

    # Extrai o ano de lançamento da data completa, se existir.
    release_year = movie_details.get("release_date") or ""

    return {
        "title": movie_details.get("title", "Sem titulo"),
        "genre": ", ".join(genres),
        "overview": movie_details.get("overview") or "",
        "cast": ", ".join(cast_names),
        "director": ", ".join(directors),
        "year": int(release_year[:4]) if release_year and release_year[:4].isdigit() else 0,
        "rating": float(movie_details.get("vote_average", 0.0) or 0.0),
    }


@dataclass
class MovieRecommender:
    # Armazena o dataframe de filmes e a matriz de similaridade calculada.
    movies: pd.DataFrame
    similarity_matrix: object

    @classmethod
    def train_from_csv(cls, csv_path: str | Path = DATA_PATH) -> "MovieRecommender":
        # Carrega o dataset CSV de filmes.
        movies = pd.read_csv(csv_path)

        # Valida se todas as colunas necessarias existem para o treino.
        required_columns = {"title", "genre", "overview", "cast", "director", "year", "rating"}
        missing = required_columns.difference(movies.columns)
        if missing:
            raise ValueError(f"Dataset incompleto. Faltam as colunas: {sorted(missing)}")

        movies = movies.copy()
        # Cria a coluna de tags textual para cada filme usando a funcao auxiliar.
        movies["tags"] = movies.apply(_join_tokens, axis=1)

        # Converte cada campo em vetores separados e aplica pesos personalizados.
        feature_matrix = _build_weighted_feature_matrix(movies, DEFAULT_FIELD_WEIGHTS)
        similarity = cosine_similarity(feature_matrix)

        return cls(movies=movies, similarity_matrix=similarity)

    @classmethod
    def train_from_tmdb(
        cls,
        tmdb_client: TMDBClient | None = None,
        page: int = 1,
        limit: int = 20,
    ) -> "MovieRecommender":
        # Usa a API da TMDB para montar um dataset a partir de filmes populares.
        client = tmdb_client or TMDBClient()
        popular_movies = client.get_popular_movies(page=page)[:limit]

        if not popular_movies:
            raise ValueError("Nenhum filme retornado pela API da TMDB para o treino do modelo.")

        # Para cada filme popular, busca detalhes completos e padroniza o registro.
        records: list[dict[str, Any]] = []
        for movie in popular_movies:
            details = client.get_movie_details(movie["id"])
            records.append(_build_tmdb_movie_record(details))

        # Cria um dataframe com os registros normalizados e gera as tags para similaridade.
        movies = pd.DataFrame(records)
        movies["tags"] = movies.apply(_join_tokens, axis=1)

        # Reutiliza a mesma logica de vetorizacao e similaridade do treino por CSV.
        feature_matrix = _build_weighted_feature_matrix(movies, DEFAULT_FIELD_WEIGHTS)
        similarity = cosine_similarity(feature_matrix)

        return cls(movies=movies, similarity_matrix=similarity)

    def save(self, model_path: str | Path = MODEL_PATH) -> Path:
        # Salva apenas os dados essenciais do modelo em um arquivo pickle.
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as file:
            # Remove a coluna temporaria de tags para economizar espaco e evitar redundancia.
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
        # Recupera o modelo salvo e reconstruye a coluna de tags para uso posterior.
        with Path(model_path).open("rb") as file:
            payload = pickle.load(file)
        movies = payload["movies"].copy()
        # Recria as tags a partir do dataframe carregado para manter a mesma estrutura.
        movies["tags"] = movies.apply(_join_tokens, axis=1)
        return cls(movies=movies, similarity_matrix=payload["similarity_matrix"])

    def get_movie(self, title: str) -> dict[str, Any]:
        # Busca um filme pelo nome e devolve um dicionario com os dados principais.
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
        # Busca filmes similares ao filme informado usando a matriz de similaridade.
        if title not in set(self.movies["title"]):
            raise ValueError(f'Filme "{title}" nao encontrado no dataset.')

        # Encontra a linha correspondente ao filme na matriz de similaridade.
        movie_index = self.movies.index[self.movies["title"] == title][0]
        distances = list(enumerate(self.similarity_matrix[movie_index]))
        ranked = sorted(distances, key=lambda item: item[1], reverse=True)

        # Remove o proprio filme da lista e retorna os proximos mais semelhantes.
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
