from typing import List

from bson import ObjectId
from pymongo.collection import Collection


class UserWatched:
    def __init__(self, users_collection: Collection):
        self.users = users_collection

    @staticmethod
    def _validate_user_id(user_id: str) -> ObjectId:
        if not user_id or not ObjectId.is_valid(user_id):
            raise ValueError("ID de usuário inválido.")

        return ObjectId(user_id)

    @staticmethod
    def _validate_movie_id(movie_id: int) -> None:
        if not isinstance(movie_id, int) or isinstance(movie_id, bool) or movie_id <= 0:
            raise ValueError("ID de filme inválido.")

    def mark_as_watched(self, user_id: str, movie_id: int) -> bool:
        user_id_obj = self._validate_user_id(user_id)
        self._validate_movie_id(movie_id)

        user = self.users.find_one({"_id": user_id_obj}, {"watched": 1})

        if not user:
            raise ValueError("Usuário não encontrado.")

        watched = user.get("watched", [])

        if movie_id in watched:
            return False

        result = self.users.update_one(
            {"_id": user_id_obj},
            {"$addToSet": {"watched": movie_id}},
        )

        if result.matched_count == 0:
            raise ValueError("Usuário não encontrado.")

        return result.modified_count > 0

    def remove_watched(self, user_id: str, movie_id: int) -> bool:
        user_id_obj = self._validate_user_id(user_id)
        self._validate_movie_id(movie_id)

        result = self.users.update_one(
            {"_id": user_id_obj},
            {"$pull": {"watched": movie_id}},
        )

        if result.matched_count == 0:
            raise ValueError("Usuário não encontrado.")

        return result.modified_count > 0

    def list_watched(self, user_id: str) -> List[int]:
        user_id_obj = self._validate_user_id(user_id)

        user = self.users.find_one({"_id": user_id_obj}, {"watched": 1})

        if not user:
            raise ValueError("Usuário não encontrado.")

        return user.get("watched", [])

    def count_watched(self, user_id: str) -> int:
        user_id_obj = self._validate_user_id(user_id)

        user = self.users.find_one({"_id": user_id_obj}, {"watched": 1})

        if not user:
            raise ValueError("Usuário não encontrado.")

        return len(user.get("watched", []))

    def is_watched(self, user_id: str, movie_id: int) -> bool:
        user_id_obj = self._validate_user_id(user_id)
        self._validate_movie_id(movie_id)

        user = self.users.find_one({"_id": user_id_obj, "watched": movie_id})

        return user is not None

    def set_initial_watched(self, user_id: str, movie_ids: List[int]) -> bool:
        user_id_obj = self._validate_user_id(user_id)

        if not isinstance(movie_ids, list):
            raise ValueError("A lista de assistidos deve ser enviada em formato de lista.")

        normalized_ids: List[int] = []
        seen = set()

        for movie_id in movie_ids:
            self._validate_movie_id(movie_id)

            if movie_id in seen:
                continue

            seen.add(movie_id)
            normalized_ids.append(movie_id)

        result = self.users.update_one(
            {"_id": user_id_obj},
            {"$set": {"watched": normalized_ids}},
        )

        if result.matched_count == 0:
            raise ValueError("Usuário não encontrado.")

        return result.modified_count > 0
