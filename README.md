# Copa dos Dados 2026

Aplicação Streamlit para analisar jogos da Copa do Mundo 2026 com base no histórico das Copas e na tabela atualizada da fase de grupos.

## O que a app faz

- Lista os 48 participantes e os 72 jogos da fase de grupos de 2026.
- Calcula um rating por seleção usando desempenho histórico em Copas, gols, vitórias, finais e títulos.
- Estima probabilidades 1X2 para cada jogo: vitória do time A, empate e vitória do time B.
- Estima gols esperados para cada seleção e total da partida.
- Calcula indicadores de gols: acima de 1,5 gol, acima de 2,5 gols, acima de 3,5 gols e ambos marcam.
- Mostra os placares mais prováveis por distribuição de Poisson.
- Mostra bandeiras dos países e um card de palpite sugerido para bolão.
- Projeta a classificação esperada do grupo com pontos, gols pró, gols contra e saldo.

## Como executar

```bash
pip install -r requirements.txt
streamlit run main.py
```

## Dados

- `Jogos Copas do Mundo.csv`: histórico de partidas usado no rating.
- `Campeoes.csv`: campeões e finalistas usados no rating.
- `data/world_cup_2026_teams.csv`: seleções, grupos e confederações da Copa 2026.
- `data/world_cup_2026_group_stage.csv`: calendário da fase de grupos 2026.

Referências usadas para a atualização de 2026:

- FIFA: calendário e sedes da Copa do Mundo 2026.
- FIFA/Wikipedia: grupos, datas e formato da fase de grupos, consultados em maio de 2026.

## Aviso

Esta aplicação é uma ferramenta estatística para estudo. Ela não garante resultados e não deve ser tratada como recomendação financeira ou promessa de lucro.
