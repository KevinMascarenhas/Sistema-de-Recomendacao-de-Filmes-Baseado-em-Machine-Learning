from typing import Any, Dict, List

from bson import ObjectId
from pymongo.collection import Collection


class UserFavorites:
    MIN_FAVORITES = 5
    MAX_FAVORITES = 20

    def __init__(self, users_collection: Collection):
        self.users = users_collection

    @staticmethod
    def _validate_user_id(user_id: str) -> ObjectId:
        if not user_id or not ObjectId.is_valid(user_id):
            raise ValueError("ID de usuário inválido.")

        return ObjectId(user_id)

    @staticmethod
    def _validate_movie_id(movie_id: int) -> int:
        if not isinstance(movie_id, int) or isinstance(movie_id, bool) or movie_id <= 0:
            raise ValueError("ID de filme inválido.")

        return movie_id

    @staticmethod
    def _validate_rating(rating: Any) -> float:
        if rating is None:
            return 0.0

        if isinstance(rating, bool):
            raise ValueError("Avaliação inválida.")

        if not isinstance(rating, (int, float)):
            raise ValueError("Avaliação deve ser um número entre 0 e 5.")

        rating_float = float(rating)
        if rating_float < 0 or rating_float > 5:
            raise ValueError("Avaliação deve ser um número entre 0 e 5.")

        return rating_float

    def _normalize_movie_ids(self, movie_ids: List[int]) -> List[int]:
        if not isinstance(movie_ids, list):
            raise ValueError("A lista de favoritos deve ser enviada em formato de lista.")

        normalized_ids: List[int] = []
        seen = set()

        for movie_id in movie_ids:
            if isinstance(movie_id, str) and movie_id.isdigit():
                movie_id = int(movie_id)

            self._validate_movie_id(movie_id)

            if movie_id in seen:
                continue

            seen.add(movie_id)
            normalized_ids.append(movie_id)

        if len(normalized_ids) < self.MIN_FAVORITES or len(normalized_ids) > self.MAX_FAVORITES:
            raise ValueError(
                f"Usuário deve definir entre {self.MIN_FAVORITES} e {self.MAX_FAVORITES} favoritos."
            )

        return normalized_ids

    def _normalize_favorite_entries(self, favorite_values: List[Any]) -> List[Dict[str, Any]]:
        if not isinstance(favorite_values, list):
            raise ValueError("A lista de favoritos deve ser enviada em formato de lista.")

        normalized_entries: List[Dict[str, Any]] = []
        seen = set()

        for value in favorite_values:
            if isinstance(value, dict):
                movie_id = value.get("movie_id")
                rating = value.get("rating", 0.0)
            else:
                movie_id = value
                rating = 0.0

            if isinstance(movie_id, str) and movie_id.isdigit():
                movie_id = int(movie_id)

            self._validate_movie_id(movie_id)
            rating = self._validate_rating(rating)

            if movie_id in seen:
                continue

            seen.add(movie_id)
            normalized_entries.append({"movie_id": movie_id, "rating": rating})

        if len(normalized_entries) < self.MIN_FAVORITES or len(normalized_entries) > self.MAX_FAVORITES:
            raise ValueError(
                f"Usuário deve definir entre {self.MIN_FAVORITES} e {self.MAX_FAVORITES} favoritos."
            )

        return normalized_entries

    def _load_user(self, user_id: str) -> dict[str, Any]:
        user_id_obj = self._validate_user_id(user_id)
        user = self.users.find_one({"_id": user_id_obj}, {"favorites": 1})

        if not user:
            raise ValueError("Usuário não encontrado.")

        return user

    def _normalize_favorite_list(self, raw_favorites: Any) -> List[Dict[str, Any]]:
        if not raw_favorites:
            return []

        if isinstance(raw_favorites, list):
            return self._normalize_favorite_entries(raw_favorites)

        raise ValueError("Formato de favoritos inválido.")

    def add_or_update_favorite(self, user_id: str, movie_id: int, rating: Any = 0.0) -> bool:
        user_id_obj = self._validate_user_id(user_id)
        movie_id = self._validate_movie_id(movie_id)
        rating = self._validate_rating(rating)

        current_entries = self.list_favorite_entries(user_id)
        updated = False

        for entry in current_entries:
            if entry["movie_id"] == movie_id:
                entry["rating"] = rating
                updated = True
                break

        if not updated:
            if len(current_entries) >= self.MAX_FAVORITES:
                raise ValueError(f"Usuário já atingiu o máximo de {self.MAX_FAVORITES} favoritos.")

            current_entries.append({"movie_id": movie_id, "rating": rating})

        result = self.users.update_one(
            {"_id": user_id_obj},
            {"$set": {"favorites": current_entries}},
        )

        if result.matched_count == 0:
            raise ValueError("Usuário não encontrado.")

        return result.modified_count > 0 or updated

    def remove_favorite(self, user_id: str, movie_id: int) -> bool:
        user_id_obj = self._validate_user_id(user_id)
        movie_id = self._validate_movie_id(movie_id)

        current_entries = self.list_favorite_entries(user_id)
        remaining_entries = [entry for entry in current_entries if entry["movie_id"] != movie_id]

        if len(remaining_entries) == len(current_entries):
            return False

        result = self.users.update_one(
            {"_id": user_id_obj},
            {"$set": {"favorites": remaining_entries}},
        )

        if result.matched_count == 0:
            raise ValueError("Usuário não encontrado.")

        return result.modified_count > 0

    def list_favorite_entries(self, user_id: str) -> List[Dict[str, Any]]:
        user = self._load_user(user_id)
        raw_favorites = user.get("favorites", [])
        return self._normalize_favorite_list(raw_favorites)

    def list_favorites(self, user_id: str) -> List[int]:
        entries = self.list_favorite_entries(user_id)
        return [entry["movie_id"] for entry in entries]

    def count_favorites(self, user_id: str) -> int:
        return len(self.list_favorite_entries(user_id))

    def is_favorite(self, user_id: str, movie_id: int) -> bool:
        movie_id = self._validate_movie_id(movie_id)
        entries = self.list_favorite_entries(user_id)
        return any(entry["movie_id"] == movie_id for entry in entries)

    def get_favorite_rating(self, user_id: str, movie_id: int) -> float | None:
        movie_id = self._validate_movie_id(movie_id)
        for entry in self.list_favorite_entries(user_id):
            if entry["movie_id"] == movie_id:
                return entry.get("rating", 0.0)
        return None

    def set_initial_favorites(self, user_id: str, movie_ids: List[int]) -> bool:
        user_id_obj = self._validate_user_id(user_id)
        normalized_ids = self._normalize_movie_ids(movie_ids)
        existing_entries = {entry["movie_id"]: entry.get("rating", 0.0) for entry in self.list_favorite_entries(user_id)}
        favorite_entries = [
            {"movie_id": movie_id, "rating": existing_entries.get(movie_id, 0.0)}
            for movie_id in normalized_ids
        ]

        result = self.users.update_one(
            {"_id": user_id_obj},
            {"$set": {"favorites": favorite_entries}},
        )

        if result.matched_count == 0:
            raise ValueError("Usuário não encontrado.")

        return result.modified_count > 0
