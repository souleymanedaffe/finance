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

@st.cache_data
def load_data(nrows=None):
    try:
        df = pd.read_excel("Retail.xlsx", nrows=nrows)
        st.write("✅ Données chargées")

        df = df.dropna(subset=['CustomerID'])
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        df['MonthPeriod'] = df['InvoiceDate'].dt.to_period('M')
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        return pd.DataFrame()

st.title('Analyse des Ventes en Ligne')
mode = st.radio("Mode de chargement", ("Rapide (5000 lignes)", "Complet (toutes les données)"), index=0)
if mode == "Rapide (5000 lignes)":
    df = load_data(nrows=5000)
else:
    with st.spinner("Chargement de toutes les données..."):
        df = load_data()

st.write(f"{len(df)} lignes chargées.")
if st.checkbox("Voir un aperçu des données"):
    st.dataframe(df.head())
