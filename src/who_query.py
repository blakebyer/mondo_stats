import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import json
import streamlit as st

@st.cache_resource(show_spinner="Fetching WHO data...")
def get_who_data(proportion="incidence"):
    url = "https://ghoapi.azureedge.net/api/Indicator"
    params = {
        "$filter":f"contains(IndicatorName,'{proportion.lower()}')",
    }
    p = requests.get(url, params)
    print(p.url)
    proportion_ind = p.json()['value']
    indicator_df = pd.DataFrame(proportion_ind)
    
    c = requests.get("https://ghoapi.azureedge.net/api/DIMENSION/COUNTRY/DimensionValues")
    countries = c.json()['value']
    country_df = pd.DataFrame(countries)
    country_df = country_df.rename(columns={'Code': 'SpatialDim'})

    all_proportion = {}

    for code in tqdm(indicator_df['IndicatorCode'], desc=f"Downloading {proportion} indicators"):
        url = f"https://ghoapi.azureedge.net/api/{code}"
        r = requests.get(url, timeout=30)

        if r.status_code == 200:
            res_df = pd.DataFrame(r.json()['value'])
            res_df['IndicatorCode'] = code
            merged = res_df.merge(indicator_df[['IndicatorCode', 'IndicatorName']], on='IndicatorCode', how='left')
            all_proportion[code] = merged

        else:
            print(f"{code}: HTTP {r.status_code}")

    non_empty = [df for df in all_proportion.values() if not df.empty]

    all_df = pd.concat(non_empty, ignore_index=True)

    return all_df