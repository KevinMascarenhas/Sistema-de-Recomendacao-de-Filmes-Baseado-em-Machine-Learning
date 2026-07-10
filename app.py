from pathlib import Path
import sys

import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))

from tmdb_client import TMDBClient


st.set_page_config(page_title="MovieMatch", layout="wide")
st.title("MovieMatch")
st.caption("Busque filmes na TMDB e receba recomendacoes da propria TMDB.")


@st.cache_resource
def load_tmdb_client() -> TMDBClient:
    return TMDBClient()


def render_movie_card(tmdb: TMDBClient, movie: dict) -> None:
    title = movie.get("title") or "Sem titulo"
    release_date = movie.get("release_date") or "Ano não informado"
    rating = float(movie.get("vote_average", 0))
    overview = movie.get("overview") or "Sem sinopse disponível."
    poster_url = tmdb.build_image_url(movie.get("poster_path"), size="w500")

    if poster_url:
        st.image(poster_url, use_container_width=True)
    st.markdown(f"### {title}")
    st.caption(f"Lançamento: {release_date} | Nota TMDB: {rating:.1f}")
    st.write(overview)


try:
    tmdb = load_tmdb_client()
except Exception as exc:
    st.error(f"Erro ao inicializar a aplicação: {exc}")
    st.stop()


with st.sidebar:
    st.header("Modo atual")
    st.write("Este app agora usa somente a TMDB (The Movie Database) para busca, detalhes e recomendações.")

query = st.text_input("Digite o nome do filme para buscar na TMDB")
results = tmdb.search_movies(query) if query else []

options = {
    f"{movie['title']} ({str(movie.get('release_date', ''))[:4]})": movie["id"]
    for movie in results
    if movie.get("title")
}

selected_label = st.selectbox("Resultados encontrados", list(options.keys())) if options else None

if selected_label:
    movie_id = options[selected_label]
    details = tmdb.get_movie_details(movie_id)

    title = details.get("title", "Sem titulo")
    overview = details.get("overview") or "Sem sinopse disponível."
    poster_url = tmdb.build_image_url(details.get("poster_path"), size="w500")
    release_date = details.get("release_date") or "Ano não informado"
    rating = float(details.get("vote_average", 0))
    genres = ", ".join(genre["name"] for genre in details.get("genres", [])) or "Generos não informados"

    details_left, details_right = st.columns([1, 1.4])
    with details_left:
        if poster_url:
            st.image(poster_url, use_container_width=True)
    with details_right:
        st.subheader(title)
        st.write(f"Lançamento: {release_date} | Nota TMDB: {rating:.1f}")
        st.write(f"Gêneros: {genres}")
        st.write(overview)

        credits = details.get("credits", {})
        cast_names = [person["name"] for person in credits.get("cast", [])[:5]]
        directors = [person["name"] for person in credits.get("crew", []) if person.get("job") == "Director"]
        st.write("Elenco principal:", ", ".join(cast_names) if cast_names else "Não informado")
        st.write("Direção:", ", ".join(directors) if directors else "Não informado")

    recommendations = tmdb.get_movie_recommendations(movie_id)
    similar_movies = tmdb.get_similar_movies(movie_id)

    if recommendations:
        st.subheader("Filmes similares:")
        recommendation_columns = st.columns(3)
        for index, movie in enumerate(recommendations[:6]):
            with recommendation_columns[index % 3]:
                render_movie_card(tmdb, movie)
    else:
        st.info("A TMDB não retornou recomendacoes para esse filme.")

    if similar_movies:
        st.subheader("Você pode gostar também:")
        similar_columns = st.columns(3)
        for index, movie in enumerate(similar_movies[:6]):
            with similar_columns[index % 3]:
                render_movie_card(tmdb, movie)
elif query:
    st.info("Nenhum filme encontrado para essa busca.")
else:
    st.info("Digite um título para buscar filmes.")
