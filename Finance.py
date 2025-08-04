import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import folium
from streamlit_folium import st_folium
import requests
import plotly.express as px

# Chargement et nettoyage des données
@st.cache_data
def load_data():
    df = pd.read_excel('Retail.xlsx')
    df = df.dropna(subset=['CustomerID'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['MonthPeriod'] = df['InvoiceDate'].dt.to_period('M')
    return df

# Chargement
df = load_data()

st.title('Analyse des Ventes en Ligne')

# 1. Modèle de Prédiction des Ventes
st.header('Modèle de Prédiction des Ventes')
monthly = (
    df.groupby(['MonthPeriod', 'StockCode'])['Quantity']
      .sum()
      .reset_index()
)
monthly['StockCode'] = monthly['StockCode'].astype(str)
le = LabelEncoder()
monthly['ProdEnc'] = le.fit_transform(monthly['StockCode'])
monthly['MonthNum'] = monthly['MonthPeriod'].dt.month
X = monthly[['ProdEnc', 'MonthNum']]
y = monthly['Quantity']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
r2 = model.score(X_test, y_test)
st.write(f"Score R² du modèle : {r2:.3f}")
if st.checkbox('Afficher Prédit vs Réel'):
    y_pred = model.predict(X_test)
    fig_scatter = px.scatter(
        x=y_test, y=y_pred,
        labels={'x':'Quantité Réelle', 'y':'Quantité Prédite'},
        title='Quantité Réelle vs Quantité Prédite',
        trendline='ols'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Section Visualisation
st.header('Visualisation des Ventes')

# 2. Évolution mensuelle (tableau)
st.subheader('Évolution mensuelle des ventes')
month_map = {1:'janvier',2:'février',3:'mars',4:'avril',5:'mai',6:'juin',7:'juillet',
             8:'août',9:'septembre',10:'octobre',11:'novembre',12:'décembre'}
rev = (
    df.groupby('MonthPeriod')['TotalPrice']
      .sum()
      .reset_index()
)
rev['TotalVente'] = rev['TotalPrice'].round(2)
rev['VentePrécédente'] = rev['TotalVente'].shift(1).fillna(0).round(2)
rev['Mois'] = rev['MonthPeriod'].dt.month.map(month_map)
rev['Évolution (%)'] = np.where(
    rev['VentePrécédente']>0,
    ((rev['TotalVente']-rev['VentePrécédente'])/rev['VentePrécédente']*100).round(2),
    0
)
df_evo = rev[['Mois','TotalVente','VentePrécédente','Évolution (%)']]
st.dataframe(df_evo, use_container_width=True)

# 3. Performance des produits
st.subheader('Performance des ventes par produit')
product_perf = (
    df.groupby('StockCode')['Quantity']
      .sum()
      .reset_index()
)
top_n = st.slider('Top N produits', 5, 20, 10)
top_prod = product_perf.nlargest(top_n,'Quantity')
fig_bar_prod = px.bar(
    top_prod, x='StockCode', y='Quantity',
    labels={'StockCode':'Produit','Quantity':'Quantité vendue'},
    title=f'Top {top_n} produits',
    hover_data=['Quantity']
)
st.plotly_chart(fig_bar_prod, use_container_width=True)

# 4. Proportion des ventes mensuelles
st.subheader('Proportion des ventes')
last = df['MonthPeriod'].max()
prev = last - 1
vals = {
    'Précédent': df[df['MonthPeriod']==prev]['Quantity'].sum(),
    'Actuel': df[df['MonthPeriod']==last]['Quantity'].sum(),
    'Autres': df['Quantity'].sum() - df[df['MonthPeriod'].isin([prev,last])]['Quantity'].sum()
}
fig_pie = px.pie(
    names=list(vals.keys()), values=list(vals.values()),
    title='Répartition des ventes',
    hole=0.3
)
st.plotly_chart(fig_pie, use_container_width=True)


# 6. Montant dépensé par client


# Calcul du montant total par client
client_amount = (
    df.groupby(['CustomerID', 'Country'])['TotalPrice']
      .sum()
      .reset_index()
      .sort_values('TotalPrice', ascending=False)
)

# Transformation de l'ID client en chaîne (pour affichage complet)
client_amount['CustomerID'] = client_amount['CustomerID'].astype(str)



# Sélection du top N clients
top_cust = st.slider('Top N clients', 5, 20, 10, key='cust2')
top_amt = client_amount.head(top_cust)

# Formatage du montant
top_amt['Montant (€)'] = top_amt['TotalPrice'].apply(lambda x: f"{x:,.0f} €".replace(',', ' '))

# Création du graphique
fig_bar_amt = px.bar(
    top_amt, x='CustomerID', y='TotalPrice',
    labels={'TotalPrice': 'Montant (€)', 'CustomerID': 'Client'},
    title=f'Top {top_cust} clients',
    hover_data={'TotalPrice': False, 'Montant (€)': True, 'Country': True}
)

# Affichage du graphique
st.plotly_chart(fig_bar_amt, use_container_width=True)

# Affichage du tableau
st.dataframe(top_amt[['CustomerID', 'Montant (€)', 'Country']])




# 7. Détails d'un client
st.header('Détails d\'un ou plusieurs clients')
ids = sorted(df['CustomerID'].unique())
cid = st.selectbox('Sélectionnez un Client', ids)
cdf = df[df['CustomerID']==cid]
st.write('Ville :', cdf['City'].mode()[0] if 'City' in cdf else 'N/A')
st.write('Pays :', cdf['Country'].mode()[0])
# Affichage des mois disponibles
periods = sorted(cdf['MonthPeriod'].unique(), reverse=True)
timestamps = [p.to_timestamp() for p in periods]
labels = [ts.strftime('%B %Y') for ts in timestamps]
# Mapping étiquette → période
label_to_period = dict(zip(labels, periods))
# Multiselect pour permettre plusieurs mois
selected_labels = st.multiselect('Mois (choisissez un ou plusieurs)', labels, default=[labels[0]])
if selected_labels:
    selected_periods = [label_to_period[l] for l in selected_labels]
    df_sel = cdf[cdf['MonthPeriod'].isin(selected_periods)]
    st.write(f"Produits achetés pour les mois sélectionnés :")
    st.dataframe(
        df_sel[['InvoiceDate','Description','Quantity','TotalPrice']]
          .rename(columns={
              'InvoiceDate':'Date',
              'Description':'Produit',
              'Quantity':'Quantité',
              'TotalPrice':'Montant (€)'
          }),
        use_container_width=True
    )
else:
    st.write("Aucun mois sélectionné.")


# Carte choroplèthe avec tooltips

clients_country = (
    df.groupby('Country')['CustomerID']
      .nunique()
      .reset_index(name='NbClients')
)


st.subheader('Carte des clients par pays')
url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/world-countries.json'
geo = requests.get(url).json()
for feat in geo['features']:
    name = feat['properties']['name']
    match = clients_country.loc[clients_country['Country']==name,'NbClients']
    feat['properties']['NbClients'] = int(match.values[0]) if not match.empty else 0
m = folium.Map(location=[20,0],zoom_start=2)
folium.Choropleth(
    geo_data=geo, data=clients_country,
    columns=['Country','NbClients'], key_on='feature.properties.name',
    fill_color='YlOrRd', nan_fill_color='white', legend_name='Nbr clients'
).add_to(m)
folium.GeoJson(
    geo,
    style_function=lambda x: {'fillColor':'transparent','color':'transparent','weight':0},
    tooltip=folium.features.GeoJsonTooltip(
        fields=['name','NbClients'], aliases=['Pays','Nbr clients'], localize=True
    )
).add_to(m)
st_folium(m, width=700, height=450)

# Signature
st.markdown("""
<div class="footer">
    Réalisé par <strong>SOULEYMANE DAFFE - DATA SCIENTIST</strong>
</div>
""", unsafe_allow_html=True)


