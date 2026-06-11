# Copa dos Dados 2026

Aplicação Streamlit para analisar jogos da Copa do Mundo 2026 com base no histórico das Copas e na tabela atualizada da fase de grupos.

## O que a app faz

- Lista os 48 participantes e os 72 jogos da fase de grupos de 2026.
- Treina uma rede neural MLP com os jogos históricos das Copas, usando apenas informações disponíveis antes de cada partida.
- Estima os gols de cada seleção com a rede neural.
- Executa 40 mil simulações de Monte Carlo por confronto para calcular probabilidades 1X2, placares e mercados de gols.
- Executa 12 mil simulações por grupo para projetar pontos, saldo e probabilidade de classificação.
- Calcula indicadores de gols: acima de 1,5 gol, acima de 2,5 gols, acima de 3,5 gols e ambos marcam.
- Mostra os placares mais prováveis observados nas simulações de Monte Carlo.
- Mostra bandeiras dos países e um card de palpite sugerido para bolão.
- Projeta a classificação esperada do grupo com pontos, gols pró, gols contra e saldo.
- Inclui a página `Confrontos diretos`, que mostra quando duas seleções já se enfrentaram em Copas, o placar e quem venceu.
- Inclui a página `Mata-mata`, que projeta a chave da Fase de 32 até a final a partir da classificação estimada dos grupos.
- Inclui a página `Estatísticas de acertos`, que compara os palpites estatísticos com os placares reais e calcula taxa de acerto.

## Como executar

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Modelo de machine learning

O motor combina:

- atributos históricos de ataque, defesa, pontos por jogo, aproveitamento e forma recente;
- uma rede neural `MLPRegressor` com duas camadas ocultas;
- validação temporal, treinando primeiro nos jogos mais antigos e avaliando nos mais recentes;
- distribuição de Poisson alimentada pelos gols previstos pela rede;
- simulação de Monte Carlo para resultados, mercados de gols, grupos e mata-mata.

## Dados

- `Jogos Copas do Mundo.csv`: histórico de partidas usado no rating.
- `Campeoes.csv`: campeões e finalistas usados no rating.
- `data/world_cup_2026_teams.csv`: seleções, grupos e confederações da Copa 2026.
- `data/world_cup_2026_group_stage.csv`: calendário da fase de grupos 2026.

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

Referências usadas para a atualização de 2026:

- FIFA: calendário e sedes da Copa do Mundo 2026.
- FIFA/Wikipedia: grupos, datas e formato da fase de grupos, consultados em maio de 2026.

## Aviso

Esta aplicação é uma ferramenta estatística para estudo. Ela não garante resultados e não deve ser tratada como recomendação financeira ou promessa de lucro.
