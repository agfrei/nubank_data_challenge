# Comentários para o revisor:
Ainda estou trabalhando no projeto enquanto essa revisão acontece.

- Os entregáveis estão na pasta `deliverable`, exceto os pdfs solicitados
- A análise exploratória vai sar baseada no notebook `notebooks/credit_risk_analysis.ipynb`
- O modelo de gastos não está prevendo nada :( Fiz por último e bem rápido pra ter um baseline, preferi enviar mesmo assim para ter uma revisão geral do projeto (organização, análise, linha de pensamento). Enquanto você revisa eu estou focado em melhorar os modelos.
- Abaixo tem uma descrição da estrutura das pastas, usei apenas uma fração:
    - os códigos estão em `src/models`
    - em `src/features` está o preprocessamento para criação de novas features, mas o código precisa de revisão, está bem bagunçado ainda
    - os `notebooks`
    - os modelos treinados estão salvos em `models` com o log de treino em `models/logs`
    - o `deliverable`
    - os dados fornecidos estão em `data/raw` e os processados com nova features estão em `data/interim`
        - A pasta `data/processed` deve ter o modelo gerado para o treino, com apenas as features usadas, mas ainda não estou salvando lá

## Executando o código:
- `make train`: executa todos os testes
    - os modelos podem ser executados individualmente com `make train_default`, `make train_fraud`, `make train_spend`
- `make predict`: executa os predicts baseado nos dados de teste e salva os arquivos na pasta `deliverable`
    - os modelos podem ser executados individualmente com `make predict_default`, `make predict_fraud`, `make predict_spend`
    - caso treine um modelo novo, para fazer o predict baseado nesse modelo tem que atualizar o arquivo de predict com o nome do modelo gerado `src/models/predic_*.py`
- o requirements.txt não está completo :( Vou acertar antes da entrega final

## Minha lista de TODO:
- separar os k-folds ordenados no tempo
- organizar a função de criação de features nos dados
- pipeline
- oversample e balanceamento das classes
- melhorar os modelos (esse ainda são apenas baselines)
- analisar matriz de confusão
- melhorar notebooks
- finalizar a documentação do código (docstrings) (ainda está em estágio inicial)
- criar environment para facilitar a reprodução do código
- ajustar os erros do lint (apesar de passar o yapf)
- melhorar a análise exploratória
    - deixar mais próxima desse modelo: https://s3.amazonaws.com/content.udacity-data.com/courses/ud651/diamondsExample_2016-05.html
    - colocar as variáveis calculadas novas
    - remover alguns gráficos que não fazem sentido
    - melhorar o aspecto visual

## Fluxo de aprovação (resultado final - problema de negócio):
- filtro de fraude (eliminatório acima de X%)
- filtro de default (eliminatório acima de X%)
- ordenar por probabilidade de default (crescente) e propensão de gasto (decrescente)
- aprovar os primeiros cuja soma do limite + custo de emissão fiquem dentro de um teto mensal estipulado
    - o teto vai sair da análise, vi que a emissão de cartões é quase uma constante mensal

## Dúvidas:
- O fluxo de aprovação proposto está aderente?
- O meu spend_score reperesenta o limite inicial sugerido para o aplicante, não ficou claro se é esse o esperado, está correto?
- O conteúdo do notebook está aderente ao esperado pela análise?
- Como está a organização do projeto? (referência ao final desse readme)
- Prioridade: melhorar o modelo ou melhorar a análise? (Estou focado em melhorar o modelo até essa resposta chegar)

Fim
==============================


nubank-data-challenge
==============================

Nubank data challenge (Udacity)

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
