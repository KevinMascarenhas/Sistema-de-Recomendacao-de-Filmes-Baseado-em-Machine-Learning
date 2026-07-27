import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


load_dotenv()


MONGODB_URI_ENV = "MONGODB_URI"
MONGODB_DATABASE_ENV = "MONGODB_DATABASE"
DEFAULT_DATABASE_NAME = "movie_recommendation"


@lru_cache(maxsize=1)   # Decorator que armazena em cache o resultado da função para evitar múltiplas conexões com o banco de dados. O cache é limitado a 1 item, garantindo que apenas uma instância do cliente MongoDB seja criada durante a execução do aplicativo.
def get_mongo_client() -> MongoClient:
    mongodb_uri = os.getenv(MONGODB_URI_ENV)

    if not mongodb_uri:
        raise ValueError(f"Variável de ambiente {MONGODB_URI_ENV} não encontrada.")

    return MongoClient(mongodb_uri)


def get_database(database_name: Optional[str] = None) -> Database:
    selected_database = (
        database_name
        or os.getenv(MONGODB_DATABASE_ENV)
        or DEFAULT_DATABASE_NAME
    )

    return get_mongo_client()[selected_database]


def get_collection(collection_name: str, database_name: Optional[str] = None) -> Collection:
    if not collection_name:
        raise ValueError("Nome da coleção é obrigatório.")

    return get_database(database_name)[collection_name]
