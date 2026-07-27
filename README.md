# MovieMatch

Aplicação web de descoberta de filmes com `Python`, `Streamlit` e integração com a API do TMDB.

A aplicação usa a TMDB como fonte principal para busca, detalhes, pôsteres, elenco e recomendações. O projeto também mantém um código de recomendador local como referência, mas o fluxo principal atual prioriza a base de dados do TMDB.

## O que o app faz

- busca filmes diretamente na TMDB (The Movie Database);
- exibe detalhes completos do filme selecionado;
- mostra pôster, sinopse, nota, elenco principal e direção;
- carrega recomendações oficiais e filmes similares do TMDB;
- permite o usuário salvar filmes favoritos e marcar filmes como assistidos;
- oferece recomendações locais como complemento, mas a fonte principal é o TMDB.

## Stack usada

- `Python`
- `Streamlit`
- `scikit-learn`
- `pandas`
- `TMDB API`
- `MongoDB`
- `pymongo`
- `bcrypt`

## Estrutura do projeto

```text
|-- app.py
|-- README.md
|-- requirements.txt
|-- .env
|-- data/
|   -- movies.csv
|-- artifacts/
|-- src/
|   |-- database.py
|   |-- recommender.py
|   |-- train_model.py
|   |-- integrations/
|   |   -- tmdb_client.py
|   |-- users/
|       |-- user.py
|       |-- user_favorites.py
|       |-- user_watched.py
```

## Arquivos principais

- `app.py`: interface Streamlit e fluxo principal da aplicação.
- `src/integrations/tmdb_client.py`: cliente responsável por autenticar e consumir a API do TMDB.
- `src/recommender.py`: código do recomendador local que gera sugestões a partir de um dataset CSV.
- `src/train_model.py`: script de treino do recomendador local.
- `src/users/user.py`, `src/users/user_favorites.py`, `src/users/user_watched.py`: serviços de usuário, favoritos e assistidos.

## Configuração

Crie um arquivo `.env` na raiz do projeto com uma destas opções:

```env
TMDB_API_READ_ACCESS_TOKEN=seu_token_aqui
```

ou

```env
API_KEY=sua_api_key_aqui
```

O app prioriza `TMDB_API_READ_ACCESS_TOKEN`, mas aceita fallback com `API_KEY`.

## Como executar

1. Crie e ative um ambiente virtual.
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute a aplicação:

```bash
streamlit run app.py
```

## Como funciona

1. O usuário digita o nome de um filme.
2. O app consulta a TMDB e mostra os resultados.
3. O usuário escolhe um filme.
4. O app carrega os detalhes do filme, incluindo créditos e sinopse em português quando disponível.
5. O app exibe recomendações oficiais do TMDB e filmes similares.
6. O app também mostra recomendações locais como complemento, quando o modelo estiver disponível.

## Endpoints TMDB utilizados

- busca de filmes: `/search/movie`
- detalhes do filme: `/movie/{movie_id}`
- recomendações: `/movie/{movie_id}/recommendations`
- similares: `/movie/{movie_id}/similar`

## Observações

- A TMDB é a fonte principal de recomendações e conteúdo.
- O arquivo `data/movies.csv` e o código de recomendador local permanecem no projeto como referência, mas não são a fonte única.
- O `.env` deve ser mantido local e não versionado.
- Se alterar credenciais e o Streamlit continuar com comportamento antigo, reinicie o app.
