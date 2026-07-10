# MovieMatch

Aplicação web de descoberta de filmes com `Python`, `Streamlit` e integração com a API do TMDB.

O projeto foi inspirado por demos clássicas de recomendação de filmes em machine learning, especialmente pela ideia de transformar o sistema em uma interface simples e demonstrável. Hoje, porém, a aplicação segue um caminho proprio e usa a TMDB como fonte principal para busca, detalhes, pôsters, elenco e recomendações.

## O que o app faz

- busca filmes diretamente na TMDB (The Movie Database);
- mostra detalhes como título, data de lançamento, nota e gêneros;
- exibe pôster e sinopse do filme selecionado;
- apresenta elenco principal e direção;
- carrega recomendações oficiais da TMDB;
- mostra também uma lista de filmes similares, baseada em dados.

## Stack usada

- `Python`
- `Streamlit`
- `requests`
- `scikit-learn`
- `pandas`
- `TMDB API`

## Estrutura do projeto

```text
.
|-- app.py
|-- src/
|   |-- tmdb_client.py
|   |-- recommender.py
|   `-- train_model.py
|-- data/
|   `-- movies.csv
|-- artifacts/
|-- requirements.txt
|-- .env
`-- README.md
```

## Arquivos principais

- `app.py`: interface Streamlit e fluxo principal da aplicação.
- `src/tmdb_client.py`: cliente responsavel por autenticar e consultar a API da TMDB.
- `src/recommender.py`: código do recomendador local antigo, mantido no repositório mas fora do fluxo principal atual.
- `src/train_model.py`: script relacionado ao recomendador local antigo.

## Configuracao

Crie um arquivo `.env` na raiz do projeto com pelo menos uma destas opções:

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
2. Instale as dependencias:

```bash
pip install -r requirements.txt
```

3. Inicie a aplicação:

```bash
streamlit run app.py
```

## Como funciona

1. O usuario digita o nome de um filme.
2. O app consulta o endpoint de busca da TMDB.
3. O usuario escolhe um resultado da lista.
4. O app busca os detalhes completos do filme, incluindo `credits`.
5. Em seguida, consulta recomendações e filmes similares da própria TMDB.
6. Tudo isso e exibido na interface com pôsters e metadados.

## Endpoints TMDB utilizados

- busca de filmes: `/search/movie`
- detalhes do filme: `/movie/{movie_id}`
- recomendações: `/movie/{movie_id}/recommendations`
- similares: `/movie/{movie_id}/similar`

## Observações

- O projeto ainda contém arquivos do recomendador local anterior para referência e possivel reaproveitamento.
- O `.env` esta ignorado no `.gitignore` e não deve ser versionado.
- Se você alterar credenciais e o Streamlit continuar com comportamento antigo, reinicie o processo do app.
