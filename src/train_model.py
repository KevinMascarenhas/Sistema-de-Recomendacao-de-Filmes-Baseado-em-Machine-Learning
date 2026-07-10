from recommender import MODEL_PATH, MovieRecommender


def main() -> None:
    # Gera o modelo a partir do CSV e salva o resultado para uso posterior no app.
    recommender = MovieRecommender.train_from_csv()
    saved_path = recommender.save(MODEL_PATH)
    print(f"Modelo salvo em: {saved_path}")


if __name__ == "__main__":
    main()
