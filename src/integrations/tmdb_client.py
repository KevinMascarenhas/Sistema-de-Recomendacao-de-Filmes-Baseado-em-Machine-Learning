from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests


class TMDBClient:
    BASE_URL = "https://api.themoviedb.org/3"

    def __init__(self, token: str | None = None) -> None:
        env_values = self._load_dotenv()
        for key, value in env_values.items():
            os.environ.setdefault(key, value)
        self.token = token or os.getenv("TMDB_API_READ_ACCESS_TOKEN") or env_values.get("TMDB_API_READ_ACCESS_TOKEN")
        self.api_key = (
            os.getenv("TMDB_API_KEY")
            or os.getenv("API_KEY")
            or env_values.get("TMDB_API_KEY")
            or env_values.get("API_KEY")
        )
        if not self.token and not self.api_key:
            raise ValueError("Defina TMDB_API_READ_ACCESS_TOKEN ou TMDB_API_KEY/API_KEY no ambiente.")
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        if self.token:
            # Prioriza o bearer token, que e o fluxo recomendado pela TMDB.
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        elif self.api_key:
            # Mantem compatibilidade com projetos que usam apenas a API key v3.
            self.session.params = {"api_key": self.api_key}

    def _load_dotenv(self) -> dict[str, str]:
        candidate_paths = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parent.parent / ".env",
        ]

        env_path = next((path for path in candidate_paths if path.exists()), None)
        if env_path is None:
            return {}

        values: dict[str, str] = {}
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
        return values

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.session.get(f"{self.BASE_URL}{path}", params=params, timeout=20)
        response.raise_for_status()
        return response.json()

    def search_movies(self, query: str, language: str = "pt-BR", page: int = 1) -> list[dict[str, Any]]:
        payload = self._get(
            "/search/movie",
            {"query": query, "language": language, "page": page, "include_adult": False},
        )
        return payload.get("results", [])

    def get_movie_details(self, movie_id: int, language: str = "pt-BR") -> dict[str, Any]:
        return self._get(
            f"/movie/{movie_id}",
            {"language": language, "append_to_response": "credits"},
        )

    def get_popular_movies(self, language: str = "pt-BR", page: int = 1) -> list[dict[str, Any]]:
        payload = self._get(
            "/movie/popular",
            {"language": language, "page": page},
        )
        return payload.get("results", [])

    def get_movie_recommendations(
        self,
        movie_id: int,
        language: str = "pt-BR",
        page: int = 1,
    ) -> list[dict[str, Any]]:
        payload = self._get(
            f"/movie/{movie_id}/recommendations",
            {"language": language, "page": page},
        )
        return payload.get("results", [])

    def get_similar_movies(
        self,
        movie_id: int,
        language: str = "pt-BR",
        page: int = 1,
    ) -> list[dict[str, Any]]:
        payload = self._get(
            f"/movie/{movie_id}/similar",
            {"language": language, "page": page},
        )
        return payload.get("results", [])

    def get_configuration(self) -> dict[str, Any]:
        return self._get("/configuration")

    def build_image_url(self, file_path: str | None, size: str = "w500") -> str | None:
        if not file_path:
            return None
        return f"https://image.tmdb.org/t/p/{size}{file_path}"
