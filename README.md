# Copa dos Dados | Modelagem e simulação de jogos

Aplicação que combina histórico de Copas, previsão de gols e simulações para explorar partidas e classificação.

**Data Science** · Aplicação Streamlit

[← Data Science, IA e Automações · Thiago Oliveira](https://github.com/thiagoramos20042/thiagoramos20042) · [Catálogo de projetos](https://github.com/thiagoramos20042/thiagoramos20042/blob/main/PROJETOS.md)

## Objetivo

Reunir dados históricos, modelos estatísticos e uma interface para comparar cenários da Copa de 2026.

## O que você encontra

- Modelo MLPRegressor para estimar gols a partir de atributos históricos.
- Simulações de Monte Carlo para confrontos, grupos e mata-mata.
- Páginas de confrontos diretos e comparação de previsões com resultados.

## Tecnologias

Python · Streamlit · pandas · NumPy · scikit-learn · CSV

## Arquivos principais

| Arquivo | Conteúdo |
| --- | --- |
| [main.py](main.py) | Aplicação e motor de modelagem/simulação. |
| [data](data) | Dados da edição de 2026. |
| [Jogos Copas do Mundo.csv](Jogos%20Copas%20do%20Mundo.csv) | Histórico de partidas. |
| [copa_dos_dados.ipynb](copa_dos_dados.ipynb) | Estudo original da edição de 2022. |
| [requirements.txt](requirements.txt) | Dependências da aplicação. |

## Como explorar

Na raiz do repositório:

```bash
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

## Contexto do projeto

As probabilidades são saídas de um modelo estatístico. Os dados locais permitem explorar a aplicação; a atualização de resultados utiliza a configuração de API descrita abaixo.

## Resultados via API

A página `Estatísticas de acertos` busca os placares da Copa 2026 diretamente do placar público da ESPN, sem necessidade de chave:

```text
https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates=20260611-20260719&limit=200
```

Somente partidas marcadas como concluídas pela API são tratadas como resultados reais. O cache é renovado a cada cinco minutos e também pode ser atualizado manualmente na aplicação.

Se quiser trocar por uma API privada ou comercial, configure no Streamlit Cloud em **Settings > Secrets**:

```toml
RESULTS_API_URL = "https://sua-api.com/world-cup-2026/results"
RESULTS_API_KEY = "seu_token_opcional"
RESULTS_API_AUTH_HEADER = "Authorization"
RESULTS_API_AUTH_PREFIX = "Bearer"
```

A API deve retornar uma lista ou um objeto com uma lista em `results`, `matches`, `fixtures`, `data` ou `response`.

Formato recomendado:

```json
[
  {
    "match_id": 1,
    "actual_home_goals": 2,
    "actual_away_goals": 1,
    "status": "finalizado"
  }
]
```

---

Integra o [portfólio de Thiago Oliveira](https://github.com/thiagoramos20042).
