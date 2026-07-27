from pathlib import Path
import sys
from typing import Any

import streamlit as st
from pymongo.errors import PyMongoError

sys.path.append(str(Path(__file__).parent / "src"))

from database import get_collection
from integrations.tmdb_client import TMDBClient
from recommender import MODEL_PATH, MovieRecommender, _build_tmdb_movie_record
from users.user import UserService
from users.user_favorites import UserFavorites
from users.user_watched import UserWatched


st.set_page_config(page_title="MovieMatch", layout="wide")


@st.cache_resource
def load_tmdb_client() -> TMDBClient:
    return TMDBClient()


@st.cache_resource
def load_user_services() -> tuple[UserService, UserFavorites, UserWatched]:
    users_collection = get_collection("users")
    return (
        UserService(users_collection),
        UserFavorites(users_collection),
        UserWatched(users_collection),
    )


@st.cache_resource
def load_recommender_and_catalog() -> tuple[MovieRecommender, list[dict[str, str]]]:
    try:
        recommender = MovieRecommender.load(MODEL_PATH)
    except Exception:
        recommender = MovieRecommender.train_from_csv()
        recommender.save(MODEL_PATH)

    movies = recommender.movies.copy()
    movies.insert(0, "movie_id", range(1, len(movies) + 1))
    movie_catalog = movies.to_dict(orient="records")
    return recommender, movie_catalog


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


def _fetch_tmdb_details_by_title(tmdb: TMDBClient, title: str, year: int | None = None) -> dict[str, Any] | None:
    candidates = tmdb.search_movies(title, language="pt-BR")
    if not candidates:
        candidates = tmdb.search_movies(title, language="en-US")

    normalized_title = title.strip().lower()
    chosen: dict[str, Any] | None = None

    if year:
        matching_year = next(
            (
                candidate
                for candidate in candidates
                if str(candidate.get("release_date", ""))[:4] == str(year)
            ),
            None,
        )
        if matching_year:
            chosen = matching_year

    if chosen is None:
        chosen = next(
            (
                candidate
                for candidate in candidates
                if str(candidate.get("title", "")).strip().lower() == normalized_title
                or str(candidate.get("original_title", "")).strip().lower() == normalized_title
            ),
            None,
        )

    if chosen is None and candidates:
        chosen = candidates[0]

    if chosen:
        try:
            return tmdb.get_movie_details(chosen["id"], language="pt-BR")
        except Exception:
            return None

    return None


def _fetch_tmdb_recommendations(tmdb: TMDBClient, movie_id: int, top_n: int = 5) -> list[dict[str, Any]]:
    movie_recommendations: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    for source in (tmdb.get_movie_recommendations(movie_id), tmdb.get_similar_movies(movie_id)):
        for movie in source:
            if not movie or not movie.get("id"):
                continue
            movie_id_value = int(movie["id"])
            if movie_id_value == movie_id or movie_id_value in seen_ids:
                continue
            movie_recommendations.append(movie)
            seen_ids.add(movie_id_value)
            if len(movie_recommendations) >= top_n:
                break
        if len(movie_recommendations) >= top_n:
            break

    return movie_recommendations


def render_recommendations_for_movie(recommender: MovieRecommender, movie_details: dict[str, Any], tmdb: TMDBClient, top_n: int = 5) -> None:
    title = movie_details.get("title", "Sem titulo")
    local_recommendations: list[dict[str, Any]] = []

    if title in set(recommender.movies["title"]):
        try:
            local_recommendations = recommender.recommend(title, top_n=top_n)
        except ValueError:
            local_recommendations = []

    if local_recommendations:
        st.subheader("Recomendações do modelo local")
        cols = st.columns(min(len(local_recommendations), 3))
        for index, recommendation in enumerate(local_recommendations):
            with cols[index % len(cols)]:
                st.markdown(f"### {recommendation['title']} ({recommendation['year']})")
                st.caption(f"Nota: {recommendation['rating']:.1f}")
                st.write(recommendation["overview"])
                st.write(f"Elenco: {recommendation['cast']}")
                st.write(f"Diretor: {recommendation['director']}")
                st.divider()

    tmdb_recommendations: list[dict[str, Any]] = []
    movie_id = movie_details.get("id")
    if movie_id is not None:
        try:
            tmdb_recommendations = _fetch_tmdb_recommendations(tmdb, int(movie_id), top_n=top_n)
        except Exception:
            tmdb_recommendations = []

    if tmdb_recommendations:
        cols = st.columns(min(len(tmdb_recommendations), 3))
        for index, movie in enumerate(tmdb_recommendations):
            with cols[index % len(cols)]:
                render_movie_card(tmdb, movie)

    if not local_recommendations and not tmdb_recommendations:
        st.warning(f"Sem recomendações disponíveis para '{title}' no momento.")


def _get_tmdb_popular_movies(tmdb: TMDBClient, limit: int = 50) -> list[dict[str, str]]:
    popular: list[dict[str, str]] = []
    page = 1

    while len(popular) < limit and page <= 3:
        movies = tmdb.get_popular_movies(page=page)
        for movie in movies:
            if not movie or not movie.get("id"):
                continue
            if any(existing.get("id") == movie["id"] for existing in popular):
                continue
            popular.append(movie)
            if len(popular) >= limit:
                break
        page += 1

    return popular[:limit]


def _get_tmdb_top_rated_movies(tmdb: TMDBClient, limit: int = 50) -> list[dict[str, str]]:
    top_rated: list[dict[str, str]] = []
    page = 1

    while len(top_rated) < limit and page <= 3:
        movies = tmdb.get_top_rated_movies(page=page)
        for movie in movies:
            if not movie or not movie.get("id"):
                continue
            if any(existing.get("id") == movie["id"] for existing in top_rated):
                continue
            top_rated.append(movie)
            if len(top_rated) >= limit:
                break
        page += 1

    return top_rated[:limit]


def _movie_option_label(movie: dict[str, str]) -> str:
    title = movie.get("title") or "Sem titulo"
    release_date = str(movie.get("release_date", ""))[:4] or "N/A"
    return f"{movie['id']} - {title} ({release_date})"


def _fetch_favorite_movies_details(tmdb: TMDBClient, favorite_ids: list[int]) -> list[dict[str, str]]:
    favorites = []
    for movie_id in favorite_ids:
        try:
            favorites.append(tmdb.get_movie_details(movie_id))
        except Exception:
            continue
    return favorites


def ensure_session_state() -> None:
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = None
    if "username" not in st.session_state:
        st.session_state["username"] = None


def logout() -> None:
    st.session_state["user_id"] = None
    st.session_state["username"] = None


def render_auth_form(user_service: UserService) -> None:
    header_columns = st.columns([3, 1])
    with header_columns[0]:
        st.markdown("<h1 style='color:#4f46e5; margin-bottom: 0;'>MovieMatch</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#c7d2fe; margin-top: 0.2rem;'>Faça login, escolha seus filmes favoritos e receba recomendações inteligentes.</p>",
            unsafe_allow_html=True,
        )
    with header_columns[1]:
        if st.session_state["user_id"]:
            st.markdown("<div style='background:#1e293b; padding:16px; border-radius:12px;'>", unsafe_allow_html=True)
            st.markdown("<strong style='color:#f8fafc;'>Usuário logado</strong>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:#cbd5e1; margin: 0.3rem 0 0;'>{st.session_state['username']}</p>", unsafe_allow_html=True)
            if st.button("Sair"):
                logout()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:#f8fafc;'>Acesso</h3>", unsafe_allow_html=True)

    if st.session_state["user_id"]:
        st.markdown("---")
        return

    auth_container = st.container()
    auth_container.header("Acesso")
    mode = auth_container.radio("Já possui conta?", ("Entrar", "Cadastrar"))

    if mode == "Entrar":
        with auth_container.form("login_form", clear_on_submit=False):
            username = st.text_input("Usuário")
            password = st.text_input("Senha", type="password")
            submit = st.form_submit_button("Entrar")

            if submit:
                user = user_service.authenticate_user(username, password)
                if user:
                    st.session_state["user_id"] = str(user["_id"])
                    st.session_state["username"] = user["username"]
                    st.success("Login realizado com sucesso.")
                else:
                    st.error("Usuário ou senha inválidos.")

    else:
        with auth_container.form("register_form", clear_on_submit=False):
            username = st.text_input("Novo usuário")
            password = st.text_input("Senha", type="password")
            email = st.text_input("Email (opcional)")
            submit = st.form_submit_button("Cadastrar")

            if submit:
                try:
                    user_service.create_user(username, password, email)
                    st.success("Conta criada com sucesso. Faça login para continuar.")
                except ValueError as error:
                    st.error(str(error))
                except Exception as error:
                    st.error(f"Erro ao criar usuário: {error}")


def render_recommendations_section(
    user_id: str,
    user_service: UserService,
    user_favorites: UserFavorites,
    recommender: MovieRecommender,
    tmdb: TMDBClient,
) -> None:
    st.header("Recomendações inteligentes")

    selected_user = user_service.get_user_by_id(user_id)
    if selected_user is None:
        st.error("Usuário não encontrado. Faça login novamente.")
        return

    try:
        favorites = user_favorites.list_favorites(user_id)
    except ValueError as error:
        st.error(str(error))
        return

    if len(favorites) < UserFavorites.MIN_FAVORITES or not selected_user.get("onboarding_completed", False):
        st.warning(
            f"Defina ao menos {UserFavorites.MIN_FAVORITES} filmes favoritos na aba Favoritos para receber recomendações personalizadas."
        )
        return

    favorite_movies = _fetch_favorite_movies_details(tmdb, favorites)

    if not favorite_movies:
        st.warning("Não foi possível carregar os detalhes dos filmes favoritos. Atualize a página e tente novamente.")
        return

    for movie in favorite_movies:
        title = movie.get("title", "Sem titulo")
        st.markdown(f"### Se você gostou de {title}, talvez você goste de:")
        render_recommendations_for_movie(recommender, movie, tmdb, top_n=5)


def render_favorites_section(
    user_id: str,
    user_service: UserService,
    user_favorites: UserFavorites,
    tmdb: TMDBClient,
) -> None:
    st.header("Gerenciar favoritos")

    selected_user = user_service.get_user_by_id(user_id)
    if selected_user is None:
        st.error("Usuário não encontrado. Faça login novamente.")
        return

    try:
        favorite_entries = user_favorites.list_favorite_entries(user_id)
    except ValueError as error:
        st.error(str(error))
        return

    favorite_labels = []
    favorite_ids_seen = set()
    for entry in favorite_entries:
        try:
            movie = tmdb.get_movie_details(entry["movie_id"])
            title = movie.get("title", "Sem título")
            release_year = str(movie.get("release_date", ""))[:4] or "N/A"
            label = f"{entry['movie_id']} - {title} ({release_year})"
            favorite_labels.append(label)
            favorite_ids_seen.add(entry["movie_id"])
            st.markdown(f"- **{title}** ({release_year}) — Nota: {entry.get('rating', 0.0):.1f}/5")
        except Exception:
            label = f"{entry['movie_id']} - Sem título (N/A)"
            favorite_labels.append(label)
            st.markdown(f"- **{entry['movie_id']}** — Nota: {entry.get('rating', 0.0):.1f}/5")

    st.markdown("---")
    st.write("Use a lista abaixo para escolher até 20 filmes favoritos dentre os títulos mais bem avaliados na TMDB.")
    st.info("Esta lista traz até 50 filmes mais bem avaliados da TMDB. Você pode desmarcar qualquer título para removê-lo dos favoritos e salvar novamente.")

    top_rated_movies = _get_tmdb_top_rated_movies(tmdb, limit=50)
    movie_options = [_movie_option_label(movie) for movie in top_rated_movies]

    for label in favorite_labels:
        if label not in movie_options:
            movie_options.append(label)

    current_labels = list(favorite_labels)

    selected_options = st.multiselect(
        "Selecione seus filmes favoritos",
        movie_options,
        default=current_labels,
        help=f"Selecione entre {UserFavorites.MIN_FAVORITES} e {UserFavorites.MAX_FAVORITES} filmes favoritos.",
    )

    if st.button("Salvar favoritos"):
        selected_ids = [int(option.split(" - ", 1)[0]) for option in selected_options]
        if len(selected_ids) < UserFavorites.MIN_FAVORITES:
            st.error(f"Selecione pelo menos {UserFavorites.MIN_FAVORITES} filmes.")
        elif len(selected_ids) > UserFavorites.MAX_FAVORITES:
            st.error(f"Você pode selecionar no máximo {UserFavorites.MAX_FAVORITES} filmes.")
        else:
            try:
                user_favorites.set_initial_favorites(user_id, selected_ids)
                user_service.complete_onboarding(user_id)
                st.success("Favoritos atualizados com sucesso.")
            except ValueError as error:
                st.error(str(error))
            except Exception as error:
                st.error(f"Erro ao salvar favoritos: {error}")


def render_tmdb_search(
    user_id: str,
    user_service: UserService,
    user_favorites: UserFavorites,
    user_watched: UserWatched,
    tmdb: TMDBClient,
) -> None:
    st.header("Busca de filmes na TMDB")
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

        current_rating = user_favorites.get_favorite_rating(user_id, movie_id) or 0.0
        is_favorite = user_favorites.is_favorite(user_id, movie_id)
        rating_input = st.slider(
            "Avalie este filme",
            min_value=0.0,
            max_value=5.0,
            value=current_rating,
            step=0.5,
            format="%0.1f",
        )

        is_watched = user_watched.is_watched(user_id, movie_id)
        if is_watched:
            st.success("Você já marcou este filme como assistido.")

        action_columns = st.columns([1, 1, 1])
        with action_columns[0]:
            if st.button(
                "Adicionar aos favoritos" if not is_favorite else "Atualizar favorito / nota",
                key=f"favorite_{movie_id}",
            ):
                try:
                    user_favorites.add_or_update_favorite(user_id, movie_id, rating_input)
                    user_service.complete_onboarding(user_id)
                    st.success("Filme salvo como favorito.")
                except Exception as exc:
                    st.error(str(exc))
        with action_columns[1]:
            if is_favorite and st.button("Remover dos favoritos", key=f"remove_{movie_id}"):
                try:
                    user_favorites.remove_favorite(user_id, movie_id)
                    st.success("Filme removido dos favoritos.")
                except Exception as exc:
                    st.error(str(exc))
        with action_columns[2]:
            if st.button(
                "Marcar como assistido" if not is_watched else "Desmarcar como assistido",
                key=f"watched_{movie_id}",
            ):
                try:
                    if is_watched:
                        user_watched.remove_watched(user_id, movie_id)
                        st.success("Filme removido da lista de assistidos.")
                    else:
                        user_watched.mark_as_watched(user_id, movie_id)
                        st.success("Filme marcado como assistido.")
                except Exception as exc:
                    st.error(str(exc))

        recommendations = tmdb.get_movie_recommendations(movie_id)
        similar_movies = tmdb.get_similar_movies(movie_id)

        if recommendations:
            st.subheader("Filmes similares:")
            recommendation_columns = st.columns(3)
            for index, movie in enumerate(recommendations[:6]):
                with recommendation_columns[index % 3]:
                    render_movie_card(tmdb, movie)
        else:
            st.info("A TMDB não retornou recomendações para esse filme.")

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


def main() -> None:
    ensure_session_state()

    try:
        user_service, user_favorites, user_watched = load_user_services()
    except PyMongoError as error:
        st.error(f"Erro ao conectar ao banco de dados: {error}")
        st.stop()

    render_auth_form(user_service)

    if not st.session_state["user_id"]:
        st.info("Faça login ou cadastre uma conta para continuar.")
        return

    try:
        recommender, movie_catalog = load_recommender_and_catalog()
    except Exception as error:
        st.error(f"Erro ao carregar o modelo de recomendações: {error}")
        st.stop()

    try:
        tmdb = load_tmdb_client()
    except Exception as exc:
        st.error(f"Erro ao inicializar a API TMDB: {exc}")
        st.stop()

    render_favorites_section(
        st.session_state["user_id"],
        user_service,
        user_favorites,
        tmdb,
    )

    st.markdown("---")
    render_recommendations_section(
        st.session_state["user_id"],
        user_service,
        user_favorites,
        recommender,
        tmdb,
    )

    st.markdown("---")
    render_tmdb_search(
        st.session_state["user_id"],
        user_service,
        user_favorites,
        user_watched,
        tmdb,
    )


if __name__ == "__main__":
    main()
