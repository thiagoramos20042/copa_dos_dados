# Copa dos Dados 2026

Aplicacao Streamlit para analisar jogos da Copa do Mundo 2026 com base no historico das Copas e na tabela atualizada da fase de grupos.

## O que a app faz

- Lista os 48 participantes e os 72 jogos da fase de grupos de 2026.
- Calcula um rating por selecao usando desempenho historico em Copas, gols, vitorias, finais e titulos.
- Estima probabilidades 1X2 para cada jogo: vitoria do time A, empate e vitoria do time B.
- Estima gols esperados para cada selecao e total da partida.
- Calcula indicadores de gols: acima de 1.5, 2.5, 3.5 e ambos marcam.
- Mostra os placares mais provaveis por distribuicao de Poisson.
- Mostra bandeiras dos paises e um card de palpite sugerido para bolao.
- Projeta a classificacao esperada do grupo com pontos, gols pro, gols contra e saldo.

## Como executar

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Dados

- `Jogos Copas do Mundo.csv`: historico de partidas usado no rating.
- `Campeoes.csv`: campeoes e finalistas usados no rating.
- `data/world_cup_2026_teams.csv`: selecoes, grupos e confederacoes da Copa 2026.
- `data/world_cup_2026_group_stage.csv`: calendario da fase de grupos 2026.

Referencias usadas para a atualizacao de 2026:

- FIFA: calendario e sedes da Copa do Mundo 2026.
- FIFA/Wikipedia: grupos, datas e formato da fase de grupos, consultados em maio de 2026.

## Aviso

Esta aplicacao e uma ferramenta estatistica para estudo. Ela nao garante resultados e nao deve ser tratada como recomendacao financeira ou promessa de lucro.
