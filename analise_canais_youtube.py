!pip install isodate

# ---------- Parte 1: Coleta de Estatísticas dos Canais ---------- #

from googleapiclient.discovery import build
import pandas as pd
import seaborn as sns; sns.set(style='white')
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import isodate
from datetime import datetime, timezone

# Chave de API do YouTube - Substituir por uma Chave Válida
api_key = 'API do Youtube'

# Lista de IDs dos canais a serem analisados
channel_ids = [
    'UCplT2lzN6MHlVHHLt6so39A',
    'UC_issB-37g9lwfAA37fy2Tg',
    'UCwAa6VoM1GCg7n4s3u9FTAg',
    'UCFuIUoyHB12qpYa8Jpxoxow',
    'UCMre98RDRijOX_fvG1gnsYg',
    'UCafFexaRoRylOKdzGBU6Pgg',
    'UCUc6UwvpQfOLDE7e52-OCMw',
    'UC9XRXKkGgCXQ2oDbjI7JEBg',
    'UCsXyoSkpyZB741fkBKeEovQ',
    'UC-dRoABm-_rKxZG9m4wzR8w',
    'UC70mr11REaCqgKke7DPJoLg',
    'UCQTTe8puVKqurziI6Do-H-Q',
    'UCGXiaSPSmORM_4_b05kfMlg',
    'UCcMcmtNSSQECjKsJA1XH5MQ',
    'UCOTem2Sh4zOU3jaeE4HzJcQ',
    'UCm63tB8wsKOVvxoU4iMpS2A',
    'UCaqc3TH-ZdPw7OTIlndvSgQ',
    'UCw9mYSlqKRXI6l4vH-tAYpQ',
    'UCpOSu4F9cqSjh1OgbmOT5cQ',
    'UCLW51-XEzuOm5RwPMChHBMw',
    'UCEQ-nGDGFupHyta90z6hVNQ',
    'UCetRsdZxDQDcgVDJd6erz6g',
    'UC0BiVs5EYh57gzGVvhddjsA',
    'UCDqfUwybgEA9Hg3P32G4Uaw',
    'UCta_wMcmkNekZqTT8PM9DWg',
    'UClBrpNsTEFLbZDDMW1xiOaQ',
    'UCzOGJclZQvPVgYZIwERsf5g',
    'UCy55mSDtBT4jnP27fZjJNQw',
    'UCmjDevp9Y8r-qi-xueD3Izg',
    'UC6rcCbDzhVoIm1V7WnwPDIQ',
    'UCZuqVjayGPpDTXegMQETlCg',
    'UCrdgeUeCll2QKmqmihIgKBQ',
    'UCZZ0NTtOgsLIT4Skr6GUpAw',
    'UCqHIWCQSq0yeE-1nbcRnt2w',
    'UCdbMvobipjxi6gdr3L1PBrQ',
    'UC9iusg9ZUmiSqXmppE9J1IQ',
    'UCrWWMZ6GVOM5zqYAUI44XXg',
    'UCLsThfULvRqd47evoMVxhRQ',
    'UCzCrdOO2GLYVnNhZUvG03lg',
    'UCORZcu08VQiRCKpVGHTWwAA',
    'UCxD5EE0H7qOhRr0tIVsOZPQ',
    'UCqQn92noBhY9VKQy4xCHPsg',
    'UCU5JicSrEM5A63jkJ2QvGYw',
    'UCrWvhVmt0Qac3HgsjQK62FQ',
    'UCRzgdJSfc5RLJKrrJR06GAg',
    'UCyw2sRlaDSYLiM07oZfL7BQ',
    'UCib793mnUOhWymCh2VJKplQ',
    'UCiB5nw8hdvyEjBI6sF50C2A',
    'UCu86KZOuqv2mVM3SOFlPbCg',
    'UCripRddD4BnaMcU833ExuwA'
]

# Inicializando o cliente da API do YouTube
youtube = build('youtube', 'v3', developerKey=api_key)

def get_channel_stats(youtube, channel_ids):
    """
    Coleta estatísticas básicas dos canais a partir da API do YouTube.
    Retorna uma lista de dicionários com os dados.
    """
    all_data = []
    request = youtube.channels().list(
        part='snippet,contentDetails,statistics',
        id=','.join(channel_ids)
    )
    response = request.execute()
    for item in response['items']:
        data = dict(
            Channel_id=item['id'],
            Created_at=item['snippet']['publishedAt'],
            Channel_name=item['snippet']['title'],
            Custom_url=item['snippet'].get('customUrl', ''),
            Subscribers=item['statistics']['subscriberCount'],
            Views=item['statistics']['viewCount'],
            Total_Videos=item['statistics']['videoCount'],
            Playlist_id=item['contentDetails']['relatedPlaylists']['uploads'],
        )
        all_data.append(data)
    return all_data

# Executando a coleta e criando o DataFrame
channel_stats = get_channel_stats(youtube, channel_ids)
channel_data = pd.DataFrame(channel_stats)

# Conversão de tipos e ordenação
channel_data['Subscribers'] = pd.to_numeric(channel_data['Subscribers'])
channel_data['Views'] = pd.to_numeric(channel_data['Views'])
channel_data['Total_Videos'] = pd.to_numeric(channel_data['Total_Videos'])
channel_data['Created_at'] = pd.to_datetime(channel_data['Created_at'], format='ISO8601')
channel_data = channel_data.sort_values(by='Subscribers', ascending=False)

# ---------- Parte 2: Cálculo do Índice de Performance ---------- #

def calc_index_performance(subs, videos, views, md_subs, md_videos, md_views):
    """
    Calcula um índice de performance ponderado com base no produto entre inscritos,
    vídeos e visualizações, normalizado pelas respectivas medianas.
    """
    return (subs * videos * views) / (md_subs * md_videos * md_views)

# Calculando as medianas para normalização
md_subs = channel_data['Subscribers'].median()
md_videos = channel_data['Total_Videos'].median()
md_views = channel_data['Views'].median()

# Calculando e aplicando log10 ao índice
channel_data['Index_Performance'] = calc_index_performance(
    channel_data['Subscribers'], 
    channel_data['Total_Videos'], 
    channel_data['Views'], 
    md_subs, md_videos, md_views
)
channel_data['Index_Performance_log10'] = np.log10(channel_data['Index_Performance'])

# ---------- Parte 3: Estatísticas Descritivas ---------- #

# Exibindo estatísticas resumidas
channel_data.describe().apply(lambda s: s.apply('{0:.2f}'.format))

# ---------- Parte 4: Visualização com Boxplots ---------- #

# Preparando dados para visualização
df_boxplot_chann = channel_data[['Subscribers', 'Views', 'Total_Videos', 'Index_Performance_log10']].rename(
    columns={
        'Subscribers': 'Inscritos',
        'Views': 'Visualizações',
        'Total_Videos': 'Vídeos publicados',
        'Index_Performance_log10': 'Índice de performance'
    }
)

# Criando boxplots
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
sns.set_style('darkgrid')

for ax, feature in zip(axes.flatten(), df_boxplot_chann.columns):
    sns.boxplot(data=df_boxplot_chann, x=feature, ax=ax)
    ax.set_title(f'Boxplot {feature}')
    ax.xaxis.set_label_text(feature)
    ax.yaxis.grid(True)

plt.tight_layout()
plt.show()

# ---------- Parte 5: Ranking de Performance ---------- #

# Ordena os canais pelo índice de performance log10
channel_data_sorted = channel_data.sort_values(by='Index_Performance_log10', ascending=False)

plt.figure(figsize=(10, 12))
ax = sns.barplot(
    data=channel_data_sorted,
    x='Channel_name',
    y='Index_Performance_log10',
    color='blue'
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
plt.xlabel('Nome do canal')
plt.ylabel('Índice de performance (log10)')
plt.title('Ranking de Performance dos Canais (escala logarítmica)')
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ---------- Parte 6: Correlação entre Variáveis ---------- #

def corrfunc(x, y, ax=None, **kws):
    """
    Função para exibir o coeficiente de correlação nos gráficos de dispersão.
    """
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'R² = {r:.2f}', xy=(.4, .9), xycoords=ax.transAxes)

# Renomeando colunas para o gráfico
channel_data_corr = channel_data[['Subscribers', 'Views', 'Total_Videos', 'Index_Performance_log10']].rename(columns={
    'Subscribers': 'Inscritos',
    'Views': 'Visualizações',
    'Total_Videos': 'Vídeos',
    'Index_Performance_log10': 'Índice de performance'
})

# Criando pairplot
g = sns.pairplot(data=channel_data_corr, kind='reg', corner=True, plot_kws={'line_kws': {'color': 'red'}})
g.map_lower(corrfunc)
plt.show()

# ---------- Parte 7: Análise de um Canal Específico ---------- #

# Canal específico para análise detalhada
channel_id = 'UCU5JicSrEM5A63jkJ2QvGYw'
canal_escolhido = channel_data[channel_data['Channel_id'] == channel_id]

# Exibindo informações principais
print(canal_escolhido[['Subscribers', 'Total_Videos', 'Views', 'Index_Performance_log10']])

# ---------- Parte 8: Idade do Canal ---------- #

# Calcula idade dos canais em anos
today = datetime.now(timezone.utc)
channel_data['Canal_age_years'] = (today - channel_data['Created_at']).dt.days / 365

# Canal para exibir idade
channel_id = 'UCFuIUoyHB12qpYa8Jpxoxow'
canal_escolhido = channel_data[channel_data['Channel_id'] == channel_id]
print(canal_escolhido[['Created_at', 'Canal_age_years']])

# ---------- Parte 9: Filtro de Canais com Muitos Vídeos ---------- #

# Filtra os canais com mais de 1000 vídeos publicados
canais_1000_videos = channel_data[channel_data['Total_Videos'] > 1000][['Channel_name', 'Total_Videos']]

# Exibindo o resultado
print(canais_1000_videos.sort_values(by='Total_Videos', ascending=False))
