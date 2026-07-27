from datetime import datetime, timezone
from typing import Optional

import bcrypt
from bson import ObjectId
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError


class UserService: # Classe responsável por gerenciar operações relacionadas a usuários, como criação, autenticação e atualização de informações.
    def __init__(self, users_collection: Collection):
        self.users = users_collection    # Coleção do MongoDB onde os usuários serão salvos.
        self.users.create_index("username", unique=True) # Cria um índice único no campo "username" para impedir que dois usuários tenham o mesmo nome de usuário.

    def hash_password(self, password: str) -> bytes:
        password_bytes = password.encode("utf-8")
        salt = bcrypt.gensalt() # Gera um salt aleatório para o hash da senha. Dessa forma, mesmo que dois usuários tenham a mesma senha, os hashes serão diferentes.
        return bcrypt.hashpw(password_bytes, salt)

    def check_password(self, password: str, password_hash: bytes) -> bool:  # Compara a senha fornecida com o hash armazenado no banco de dados para autenticação do usuário.
        password_bytes = password.encode("utf-8")
        return bcrypt.checkpw(password_bytes, password_hash)
    
    def create_user(self, username: str, password: str, email: Optional[str] = None) -> dict: # dict para retornar os dados do usuário criado, sem a senha.
        if not username or not password:
            raise ValueError("Username e senha são obrigatórios.")

        password_hash = self.hash_password(password)

        user_data = {     # documento que será salvo na coleção de usuários do MongoDB.
            "username": username.strip().lower(),
            "password_hash": password_hash,
            "email": email.strip().lower() if email else None,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "is_active": True,
            "onboarding_completed": False,  # indica que o usuário ainda não selecionou 20 filmes favoritos para receber recomendações personalizadas.
        }

        try:
            result = self.users.insert_one(user_data) # insert one é uma função do pymongo que insere um documento novo na coleção. Se o username já existir, será levantada uma exceção DuplicateKeyError.
        except DuplicateKeyError:
            raise ValueError("Este username já está em uso.")

        user_data["_id"] = result.inserted_id
        user_data.pop("password_hash", None)

        return user_data

    def authenticate_user(self, username: str, password: str) -> Optional[dict]:
        user = self.users.find_one({   # find one é uma função do pymongo que retorna o primeiro documento que corresponde à consulta. 
            "username": username.strip().lower(),
            "is_active": True,
        })

        if not user:
            return None

        if not self.check_password(password, user["password_hash"]):
            return None

        user.pop("password_hash", None)
        return user

    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        if not ObjectId.is_valid(user_id):
            return None

        user = self.users.find_one({"_id": ObjectId(user_id)})

        if user:
            user.pop("password_hash", None)

        return user

    def get_user_by_username(self, username: str) -> Optional[dict]:
        user = self.users.find_one({"username": username.strip().lower()})

        if user:
            user.pop("password_hash", None)

        return user

    def complete_onboarding(self, user_id: str) -> bool:
        if not ObjectId.is_valid(user_id):
            return False

        result = self.users.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "onboarding_completed": True,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

        return result.modified_count > 0 # modified_count indica quantos documentos foram modificados. Se o número for maior que 0, significa que o usuário foi atualizado com sucesso e o onboarding foi concluído.
