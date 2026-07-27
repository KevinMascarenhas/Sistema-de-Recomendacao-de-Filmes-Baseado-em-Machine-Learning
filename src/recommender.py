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
    parts = []
    # Campos textuais padrão
    for field in ("genre", "subgenre", "overview", "cast", "director", "country"):
        if field in row and pd.notna(row[field]):
            parts.append(str(row[field]))

    # Campos numéricos que também podem contribuir como tokens
    if "year" in row and pd.notna(row["year"]):
        parts.append(str(int(row["year"])) if str(row["year"]).isdigit() else str(row["year"]))
    if "rating" in row and pd.notna(row["rating"]):
        parts.append(str(float(row["rating"])))

    return _normalize_text(" ".join(parts))


DEFAULT_FIELD_WEIGHTS: dict[str, float] = {
    "genre": 3.0,
    "subgenre": 2.5,
    "overview": 2.0,
    "cast": 1.5,
    "director": 1.0,
    "country": 1.0,
    "year": 0.5,
    "rating": 0.5,
}


def _build_weighted_feature_matrix(
    movies: pd.DataFrame,
    field_weights: dict[str, float],
) -> tuple[object, dict[str, CountVectorizer]]:
    # Converte cada campo em uma matriz de características separada.
    # Depois aplica o peso desejado em cada matriz e concatena horizontalmente.
    feature_matrices = []
    vectorizers: dict[str, CountVectorizer] = {}

    # Itera sobre os campos declarados nos pesos, mas só processa os que existirem no dataframe.
    for field in field_weights.keys():
        if field not in movies.columns:
            continue

        raw = movies[field].fillna("")
        raw_text = raw.astype(str)
        if raw_text.str.strip().eq("").all():
            continue

        vectorizer = CountVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
        try:
            matrix = vectorizer.fit_transform(raw_text)
        except ValueError as exc:
            if "empty vocabulary" in str(exc).lower():
                continue
            raise

        if matrix.shape[1] == 0:
            continue

        vectorizers[field] = vectorizer

        weight = field_weights.get(field, 1.0)
        if weight != 1.0:
            matrix = matrix.multiply(weight)

        feature_matrices.append(matrix)

    if not feature_matrices:
        raise ValueError(
            "Não foi possível construir a matriz de características. Verifique se o dataset contém pelo menos um campo de texto válido para vetorização."
        )

    return hstack(feature_matrices), vectorizers


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

    # Tenta extrair keywords/etiquetas da TMDB para usar como subgenres (palavras-chave que atuam como subgêneros)
    subgenres = []
    # A API pode retornar keywords em chaves diferentes dependendo do append_to_response
    keywords_block = movie_details.get("keywords") or movie_details.get("keyword") or {}
    # keywords_block pode ter 'keywords' ou 'results'
    keywords_list = keywords_block.get("keywords") if isinstance(keywords_block, dict) else None
    if not keywords_list:
        keywords_list = keywords_block.get("results") if isinstance(keywords_block, dict) else None

    if isinstance(keywords_list, list):
        subgenres = [k.get("name") for k in keywords_list if k.get("name")]

    # Coleta países de produção, se houver
    countries = [c.get("name") for c in movie_details.get("production_countries", [])]

    # Extrai o ano de lançamento da data completa, se existir.
    release_year = movie_details.get("release_date") or ""

    return {
        "title": movie_details.get("title", "Sem titulo"),
        "genre": ", ".join(genres),
        "subgenre": ", ".join(subgenres),
        "overview": movie_details.get("overview") or "",
        "cast": ", ".join(cast_names),
        "director": ", ".join(directors),
        "country": ", ".join([c for c in countries if c]),
        "year": int(release_year[:4]) if release_year and release_year[:4].isdigit() else 0,
        "rating": float(movie_details.get("vote_average", 0.0) or 0.0),
    }


@dataclass
class MovieRecommender:
    # Armazena o dataframe de filmes, a matriz de similaridade calculada e os vetorizadores usados.
    movies: pd.DataFrame
    similarity_matrix: object
    feature_matrix: object
    vectorizers: dict[str, CountVectorizer]
    field_weights: dict[str, float]

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
        feature_matrix, vectorizers = _build_weighted_feature_matrix(movies, DEFAULT_FIELD_WEIGHTS)
        similarity = cosine_similarity(feature_matrix)

        return cls(
            movies=movies,
            similarity_matrix=similarity,
            feature_matrix=feature_matrix,
            vectorizers=vectorizers,
            field_weights=DEFAULT_FIELD_WEIGHTS,
        )

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
        feature_matrix, vectorizers = _build_weighted_feature_matrix(movies, DEFAULT_FIELD_WEIGHTS)
        similarity = cosine_similarity(feature_matrix)

        return cls(
            movies=movies,
            similarity_matrix=similarity,
            feature_matrix=feature_matrix,
            vectorizers=vectorizers,
            field_weights=DEFAULT_FIELD_WEIGHTS,
        )

    def save(self, model_path: str | Path = MODEL_PATH) -> Path:
        # Salva apenas os dados essenciais do modelo em um arquivo pickle.
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as file:
            # Remove a coluna temporaria de tags para economizar espaco e evitar redundancia.
            payload = {
                "movies": self.movies.drop(columns=["tags"], errors="ignore"),
                "similarity_matrix": self.similarity_matrix,
                "feature_matrix": self.feature_matrix,
                "vectorizers": self.vectorizers,
                "field_weights": self.field_weights,
            }
            pickle.dump(payload, file)
        return model_path

    @classmethod
    def load(cls, model_path: str | Path = MODEL_PATH) -> "MovieRecommender":
        # Recupera o modelo salvo e reconstruye a coluna de tags para uso posterior.
        with Path(model_path).open("rb") as file:
            payload = pickle.load(file)
        movies = payload["movies"].copy()
        # Recria as tags a partir do dataframe carregado para manter a mesma estrutura.
        movies["tags"] = movies.apply(_join_tokens, axis=1)
        return cls(
            movies=movies,
            similarity_matrix=payload["similarity_matrix"],
            feature_matrix=payload["feature_matrix"],
            vectorizers=payload["vectorizers"],
            field_weights=payload["field_weights"],
        )

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

    def _vectorize_record(self, record: dict[str, Any]) -> object:
        features = []
        for field, weight in self.field_weights.items():
            if field not in record or record[field] is None:
                continue

            raw_text = str(record[field])
            vectorizer = self.vectorizers.get(field)
            if vectorizer is None:
                continue

            matrix = vectorizer.transform([raw_text])
            if weight != 1.0:
                matrix = matrix.multiply(weight)
            features.append(matrix)

        if not features:
            raise ValueError("Nenhum campo disponível para vetorização do registro externo.")

        return hstack(features)

    def _select_diverse_recommendations(
        self,
        ranked: list[tuple[int, float]],
        target_title: str,
        top_n: int,
    ) -> list[dict[str, Any]]:
        recommendations: list[dict[str, Any]] = []
        seen_titles: set[str] = set()
        seen_directors: set[str] = set()
        fallback: list[dict[str, Any]] = []
        normalized_target = target_title.strip().lower()

        for index, score in ranked:
            movie = self.movies.iloc[index]
            movie_title = str(movie.get("title", "")).strip()
            normalized_title = movie_title.lower()
            if not movie_title or normalized_title == normalized_target:
                continue
            if normalized_title in seen_titles:
                continue

            director = str(movie.get("director", "")).strip()
            recommendation = {
                "title": movie_title,
                "genre": movie["genre"],
                "year": int(movie["year"]),
                "rating": float(movie["rating"]),
                "score": round(float(score), 3),
                "overview": movie["overview"],
                "cast": movie["cast"],
                "director": movie["director"],
            }

            if not recommendations:
                recommendations.append(recommendation)
                seen_titles.add(normalized_title)
                if director:
                    seen_directors.add(director)
                continue

            if director and director not in seen_directors and len(seen_directors) < min(3, top_n):
                recommendations.append(recommendation)
                seen_titles.add(normalized_title)
                seen_directors.add(director)
            else:
                fallback.append(recommendation)

            if len(recommendations) >= top_n:
                break

        for recommendation in fallback:
            if len(recommendations) >= top_n:
                break
            normalized_title = recommendation["title"].strip().lower()
            if normalized_title in seen_titles:
                continue
            recommendations.append(recommendation)
            seen_titles.add(normalized_title)

        return recommendations

    def recommend(self, title: str, top_n: int = 5) -> list[dict[str, Any]]:
        # Busca filmes similares ao filme informado usando a matriz de similaridade.
        if title not in set(self.movies["title"]):
            raise ValueError(f'Filme "{title}" nao encontrado no dataset.')

        # Encontra a linha correspondente ao filme na matriz de similaridade.
        movie_index = self.movies.index[self.movies["title"] == title][0]
        distances = list(enumerate(self.similarity_matrix[movie_index]))
        ranked = sorted(distances, key=lambda item: item[1], reverse=True)
        return self._select_diverse_recommendations(ranked, title, top_n)

    def recommend_from_record(self, record: dict[str, Any], top_n: int = 5) -> list[dict[str, Any]]:
        record_vector = self._vectorize_record(record)
        distances = cosine_similarity(self.feature_matrix, record_vector).flatten()
        ranked = sorted(enumerate(distances), key=lambda item: item[1], reverse=True)
        target_title = str(record.get("title", "")).strip().lower()
        return self._select_diverse_recommendations(ranked, target_title, top_n)
