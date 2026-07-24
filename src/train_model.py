from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from integrations.tmdb_client import TMDBClient
from recommender import MODEL_PATH, MovieRecommender


def main() -> None:
    # Gera o modelo a partir de dados populares da TMDB e salva o resultado para uso posterior.
    tmdb_client = TMDBClient()
    recommender = MovieRecommender.train_from_tmdb(tmdb_client=tmdb_client, page=1, limit=20)
    saved_path = recommender.save(MODEL_PATH)
    print(f"Modelo salvo em: {saved_path}")


if __name__ == "__main__":
    main()
