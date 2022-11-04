import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def page1():
    # Adicionando a logo
    st.sidebar.markdown("# Análise Exploratória dos Dados")
#     # Adicionando a logo
    image = Image.open("logo.png")
    st.image(image)
    st.title("Análise Exploratória dos Dados")
    st.header("Entendimento do problema")
    st.write(
        "Construir um algortimo de Machine Learning capaz de prever quem vai ganhar a Copa do Mundo 2022")
    st.header("Coleta dos dados")
    st.write(
        "Foi disponibilizado conjuntos de dados no formato Excel e nossa tarefa foi construir um Banco de dados no MongoDB para ingestão dos dados e consultas das tabelas para as futuras análises para o projeto. As tabelas que compõem a base de dados são (df_jogoscopasdomundo, df_jogadores_copasdomundo, df_campeoes_copasdomundo)")
    st.header("Tabela jogos Copas do Mundo")

    # df jogos copas do mundo
    df_jogoscopasdomundo = pd.read_csv("Jogos Copas do Mundo.csv", encoding="cp1252 ")
    st.dataframe(df_jogoscopasdomundo)

    st.header("Tabela Jogadores Copas do Mundo")

    # df jogadores copas do mundo
    df_jogadores_copasdomundo = pd.read_csv("Jogadores.csv", encoding="cp1252 ")
    st.dataframe(df_jogadores_copasdomundo)

    st.header("Tabela Campeões Copas do Mundo")

    # df campeões copas do mundo
    df_campeoes_copasdomundo = pd.read_csv("Campeoes.csv", encoding="cp1252 ")
    st.dataframe(df_campeoes_copasdomundo)

    # Perguntas respondidas com os dados

    st.header("Insights da base de dados")
    st.subheader(
        "1 - Quem são os maiores vencedores?")
    descending_order = df_campeoes_copasdomundo["Vencedor"].value_counts().sort_values(ascending=False).index

    fig = plt.figure(figsize=(12, 6))
    plt.title("Seleções que venceram a Copa do Mundo")
    sns.countplot(data=df_campeoes_copasdomundo, x="Vencedor", order=descending_order)
    st.pyplot(fig)

    st.subheader(
        "2 - Seleções que mais ficaram em segundo lugar  na Copa do Mundo")
    df_campeoes_copasdomundo['Segundo'].replace('Germany FR', 'Germany', inplace=True)
    descending_order = df_campeoes_copasdomundo["Segundo"].value_counts().sort_values(ascending=False).index

    fig = plt.figure(figsize=(12, 6))
    plt.title("Seleções que mais ficaram em segundo lugar  na Copa do Mundo")
    sns.countplot(data=df_campeoes_copasdomundo, x="Segundo", order=descending_order)
    st.pyplot(fig)

    # Alterando o tipo da coluna de string para numérica
    df_jogoscopasdomundo[['Publico']] = df_jogoscopasdomundo[['Publico']].apply(pd.to_numeric)
    media_publico = df_jogoscopasdomundo.groupby("Ano")[["Publico"]].mean().reset_index()
    st.subheader(
        "3 - Qual foi a média de Público na Copa do Mundo")
    fig = plt.figure(figsize=(12, 6))
    plt.title("Média público Copa do mundo", color="black")
    sns.boxplot(x = df_jogoscopasdomundo["Ano"], y= df_jogoscopasdomundo["Publico"])
    st.pyplot(fig)

    # 4 Média de Gols copas do mundo
    df_jogoscopasdomundo["TotalGols"] = df_jogoscopasdomundo["GolsTimeDaCasa"] + df_jogoscopasdomundo["GolsTimeVisitante"]
    st.subheader(
        "4 - Média de Gols na Copa do Mundo")
    fig = plt.figure(figsize=(12, 6))
    plt.title("Média de Gols", color="black")
    sns.boxplot(x= df_jogoscopasdomundo["Ano"], y= df_jogoscopasdomundo["TotalGols"])
    st.pyplot(fig)


def page2():
    st.sidebar.markdown("# Predição Jogos da Copa do Mundo")
    image = Image.open("logo.png")
    st.image(image)

    st.title("Copa dos Dados")
    st.text("Algoritmo de Machine Learning capaz de prever qual time vai ganhar a Copa do Mundo 2022")

    df_selecoes = pd.read_csv("Selecoes2022.csv")

    todas_selecoes = sorted(df_selecoes['Selecoes'].unique())

    selecionar_primeira_selecao = st.selectbox('Primeira Seleção', todas_selecoes)

    selecao_b = df_selecoes[df_selecoes['Selecoes'] != selecionar_primeira_selecao]
    selecionar_segunda_selecao = st.selectbox('Segunda Seleção', selecao_b)

    model = joblib.load('model.pkl')

    nome_time = {'France': 0, 'Mexico': 1, 'USA': 2, 'Belgium': 3, 'Yugoslavia': 4, 'Brazil': 5, 'Romania': 6,
                 'Peru': 7, 'Argentina': 8,
                 'Chile': 9, 'Bolivia': 10, 'Paraguay': 11, 'Uruguay': 12, 'Austria': 13, 'Hungary': 14, 'Egypt': 15,
                 'Switzerland': 16, 'Netherlands': 17,
                 'Sweden': 18, 'Germany': 19, 'Spain': 20, 'Italy': 21, 'Czechoslovakia': 22, 'Dutch East Indies': 23,
                 'Cuba': 24, 'Norway': 25,
                 'Poland': 26, 'England': 27, 'Scotland': 28, 'Turkey': 29, 'Korea Republic': 30, 'Soviet Union': 31,
                 'Wales': 32, 'Northern Ireland': 33,
                 'Colombia': 34, 'Bulgaria': 35, 'Korea DPR': 36, 'Portugal': 37, 'Israel': 38, 'Morocco': 39,
                 'El Salvador': 40, 'German DR': 41,
                 'Australia': 42, 'Zaire': 43, 'Haiti': 44, 'Tunisia': 45, 'IR Iran': 46, 'Iran': 47, 'Cameroon': 48,
                 'New Zealand': 49, 'Algeria': 50,
                 'Honduras': 51, 'Kuwait': 52, 'Canada': 53, 'Iraq': 54, 'Denmark': 55, 'rn">United Arab Emirates': 56,
                 'Costa Rica': 57, 'rn">Republic of Ireland': 58,
                 'Saudi Arabia': 59, 'Russia': 60, 'Greece': 61, 'Nigeria': 62, 'South Africa': 63, 'Japan': 64,
                 'Jamaica': 65, 'Croatia': 66,
                 'Senegal': 67, 'Slovenia': 68, 'Ecuador': 69, 'China PR': 70, 'rn">Trinidad and Tobago': 71,
                 "Côte d'Ivoire": 72, 'rn">Serbia and Montenegro': 73,
                 'Angola': 74, 'Czech Republic': 75, 'Ghana': 76, 'Togo': 77, 'Ukraine': 78, 'Serbia': 79,
                 'Slovakia': 80, 'rn">Bosnia and Herzegovina': 81,
                 'Iceland': 82, 'Panama': 83, 'Qatar': 84, 'Iran': 85, 'South Korea': 86, 'United States': 87}

    df_campeoes = pd.read_csv("Campeoes.csv")
    campeoes = df_campeoes['Vencedor'].value_counts()

    def predicao(timeA, timeB):
        idA = nome_time[timeA]
        idB = nome_time[timeB]
        campeaoA = campeoes.get(timeA) if campeoes.get(timeA) != None else 0
        campeaoB = campeoes.get(timeB) if campeoes.get(timeB) != None else 0

        x = np.array([idA, idB, campeaoA, campeaoB]).astype('float64')
        x = np.reshape(x, (1, -1))
        _y = model.predict_proba(x)[0]

        text = (
                    'Chance de ' + timeA + ' vencer ' + timeB + ' é {}\nChance de ' + timeB + ' vencer ' + timeA + ' e {}\nChance de ' + timeA + ' e ' + timeB + ' empatar é {}').format(
            _y[1] * 100, _y[2] * 100, _y[0] * 100)
        return _y[0], text

    prob1, text1 = predicao(selecionar_primeira_selecao, selecionar_segunda_selecao)

    if st.button('Realizar predição do Jogo'):
        st.text(text1)




#Chamando as funções de cada página
page_names_to_funcs = {
    "Análise Exploratória dos Dados": page1,
    "Predição Jogos Copa do Mundo": page2,

}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
