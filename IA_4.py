pip install googletrans==4.0.0-rc1


import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


import nltk
nltk.download('vader_lexicon')


import pandas as pd
from googletrans import Translator
import time
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Lê o arquivo CSV
df = pd.read_csv('/content/Dados_Fonais_46_3.csv', sep=';', encoding='Windows-1252', header=None, names=['mes', 'ano', 'Comentario'])

# Converte a coluna 'ano' para o tipo inteiro
df['ano'] = df['ano'].astype(int)

# Cria uma instância do tradutor
translator = Translator()

# Percorre as linhas do DataFrame e traduz as linhas que não contêm os meses do ano
for i, row in df.iterrows():
    if row['Comentario'] not in ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']:
        try:
            # Traduz o comentário da linha atual
            translated = translator.translate(row['Comentario'], src='pt', dest='en').text
            # Atualiza o DataFrame com o comentário traduzido
            df.at[i, 'Comentario'] = translated
        except:
            # Imprime uma mensagem de erro caso ocorra um erro na tradução
            print(f"Erro ao traduzir a linha {i}: {row['Comentario']}")
            # Espera 10 segundos antes de tentar novamente
            time.sleep(10)

# Cria uma instância do leitor de sentimentos
sia = SentimentIntensityAnalyzer()

# Calcula o sentimento de cada comentário e adiciona uma nova coluna ao DataFrame
df['Sentimento'] = df['Comentario'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Agrupa o DataFrame por ano e calcula a média do sentimento para cada ano
df_anual = df.groupby('ano').mean().reset_index()

# Cria um gráfico de linha para mostrar a popularidade do jardim botânico ao longo dos anos
plt.plot(df_anual['ano'], df_anual['Sentimento'])
plt.xlabel('Ano')
plt.ylabel('Sentimento')
plt.title('Popularidade do Jardim Botânico')
plt.show()

# Cria gráficos de linha para mostrar a popularidade do jardim botânico por ano e por mês
for ano in df['ano'].unique():
    # Filtra as linhas do DataFrame para o ano atual
    df_ano = df[df['ano'] == ano]
    
    # Cria uma nova coluna 'mes_numerico' para ordenação pelos meses
    meses = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']
    df_ano['mes_numerico'] = df_ano['mes'].apply(lambda x: meses.index(x.lower())+1)

    # Ordena as linhas pelo mês
    df_ano = df_ano.sort_values(by='mes_numerico')
    
    # Cria um gráfico de linha para mostrar a popularidade do jardim botânico por mês
    plt.bar(df_ano['mes'], df_ano['Sentimento'].cumsum())
    plt.xlabel('Mês')
    plt.ylabel('Sentimento acumulado')
    plt.title(f'Popularidade do Jardim Botânico em {ano}')
    plt.show()


