# MovieMatch

Projeto de recomendação de filmes com `Python`, `scikit-learn` e `Streamlit`.

A ideia se inspira no repositório `entbappy/Movie-Recommender-System-Using-Machine-Learning`, principalmente no formato de app demonstrável e no uso de recomendação por similaridade. Ao mesmo tempo, este projeto segue um caminho próprio:

- código menor e mais fácil de entender;
- pipeline local sem dependência obrigatória de API externa;
- base CSV simples para estudo e evolucao;
- interface Streamlit com foco em clareza.

## O que o sistema faz

- carrega uma base com mais de 50 filmes;
- combina `genre`, `overview`, `cast` e `director` em uma representação textual;
- transforma esse texto em vetores com `CountVectorizer`;
- calcula proximidade entre filmes com `cosine_similarity`;
- exibe recomendações em uma interface web.

## Estrutura do projeto

```text
.
|-- app.py
|-- data/
|   `-- movies.csv
|-- artifacts/
|   `-- movie_recommender.pkl
|-- src/
|   |-- recommender.py
|   `-- train_model.py
|-- outputs/
|-- work/
|-- requirements.txt
`-- README.md
```

## Como executar

1. Crie e ative um ambiente virtual.
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Gere o artefato do modelo:

```bash
python src/train_model.py
```

4. Rode a interface:

```bash
streamlit run app.py
```

## Diferencas em relação ao projeto de referencia

- usa uma estrutura mais enxuta;
- nao depende de chave TMDB para funcionar;
- trabalha com um dataset local controlado;
- deixa o fluxo de treino e carga mais explicito;
- permite evoluir para API externa depois, sem reescrever a base.

## Proximos passos sugeridos

- integrar posters reais via TMDB;
- adicionar busca aproximada por titulo;
- exibir justificativas simples da recomendacao, como elenco ou diretor em comum;
- separar preprocessamento, treino e interface em modulos ainda mais claros;
- evoluir para um sistema hibrido com preferencias de usuarios.
