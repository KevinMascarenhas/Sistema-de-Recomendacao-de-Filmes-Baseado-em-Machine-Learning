from typing import List

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
    def _validate_movie_id(movie_id: int) -> None:
        if not isinstance(movie_id, int) or isinstance(movie_id, bool) or movie_id <= 0:
            raise ValueError("ID de filme inválido.")

    def _normalize_movie_ids(self, movie_ids: List[int]) -> List[int]:
        if not isinstance(movie_ids, list):
            raise ValueError("A lista de favoritos deve ser enviada em formato de lista.")

        normalized_ids: List[int] = [] 
        seen = set() # seen é um conjunto que armazena os IDs de filmes já processados para evitar duplicatas.

        for movie_id in movie_ids:
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

    def add_favorite(self, user_id: str, movie_id: int) -> bool:
        user_id_obj = self._validate_user_id(user_id)
        self._validate_movie_id(movie_id)

        user = self.users.find_one({"_id": user_id_obj}, {"favorites": 1}) # find_one é uma função do pymongo que retorna o primeiro documento que corresponde à consulta. Neste caso, ele busca o usuário pelo ID e retorna apenas o campo "favorites" do documento. favorites: 1 porque indica que queremos incluir somente o campo "favorites" no resultado da consulta.

        if not user:
            raise ValueError("Usuário não encontrado.")

        favorites = user.get("favorites", [])

        if len(favorites) >= self.MAX_FAVORITES:
            raise ValueError(f"Usuário já atingiu o máximo de {self.MAX_FAVORITES} favoritos.")

        if movie_id in favorites:
            return False

        result = self.users.update_one(
            {"_id": user_id_obj},
            {"$addToSet": {"favorites": movie_id}}, # addToSet é um operador do MongoDB que adiciona um valor a um array somente se ele ainda não estiver presente. Neste caso, ele adiciona o movie_id ao array favorites do usuário, garantindo que não haja duplicatas.
        )

        if result.matched_count == 0:
            raise ValueError("Usuário não encontrado.")

        return result.modified_count > 0

    def remove_favorite(self, user_id: str, movie_id: int) -> bool:
        user_id_obj = self._validate_user_id(user_id)
        self._validate_movie_id(movie_id)

        result = self.users.update_one(
            {"_id": user_id_obj},
            {"$pull": {"favorites": movie_id}}, # pull é um operador do MongoDB que remove todos os elementos de um array que correspondem a uma condição especificada. Neste caso, ele remove o movie_id do array favorites do usuário.
        )

        if result.matched_count == 0:
            raise ValueError("Usuário não encontrado.")

        return result.modified_count > 0

    def list_favorites(self, user_id: str) -> List[int]:
        user_id_obj = self._validate_user_id(user_id)

        user = self.users.find_one({"_id": user_id_obj}, {"favorites": 1})

        if not user:
            raise ValueError("Usuário não encontrado.")

        return user.get("favorites", [])

    def count_favorites(self, user_id: str) -> int:
        user_id_obj = self._validate_user_id(user_id)

        user = self.users.find_one({"_id": user_id_obj}, {"favorites": 1})

        if not user:
            raise ValueError("Usuário não encontrado.")

        return len(user.get("favorites", []))

    def is_favorite(self, user_id: str, movie_id: int) -> bool:
        user_id_obj = self._validate_user_id(user_id)
        self._validate_movie_id(movie_id) 

        user = self.users.find_one({"_id": user_id_obj, "favorites": movie_id}) 

        return user is not None

    def set_initial_favorites(self, user_id: str, movie_ids: List[int]) -> bool:
        user_id_obj = self._validate_user_id(user_id)
        normalized_ids = self._normalize_movie_ids(movie_ids)

        result = self.users.update_one(
            {"_id": user_id_obj},
            {"$set": {"favorites": normalized_ids}}, # set é um operador do MongoDB que substitui o valor de um campo específico em um documento. Neste caso, ele define o campo "favorites" do usuário com o array normalized_ids, que contém os IDs de filmes normalizados e validados.
        )

        if result.matched_count == 0:
            raise ValueError("Usuário não encontrado.")

        return result.modified_count > 0
