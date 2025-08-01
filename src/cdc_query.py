import os
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

CDC_APP_TOKEN = os.getenv("CDC_APP_TOKEN")
CDC_API_KEY = os.getenv("CDC_API_KEY")

@st.cache_resource(show_spinner="Fetching CDC data...")
def get_cdc_data(proportion="prevalence", limit=10000):
    url = "https://data.cdc.gov/resource/hksd-2xuw.json"
    params = {
        "$where":f"LOWER(datavaluetype) LIKE '%{proportion.lower()}%'",
        "$limit":limit,
        "$$app_token":CDC_APP_TOKEN
    }
    r = requests.get(url, params=params)
    results_df = pd.DataFrame(r.json())
    return results_df