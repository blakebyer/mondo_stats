import streamlit as st
import cdc_agent
import who_agent
import pandas as pd

st.set_page_config(layout="wide")
st.title("MONDO Disease Statistics")

options = ["WHO", "CDC"]
data_choice = st.segmented_control(
    "Select data source:", options, selection_mode="single"
    )

if data_choice == "WHO":
    statistic = ["Incidence", "Prevalence"]
    prop_choice = st.segmented_control(
        "Selected statistic:", statistic, default=statistic[0]
    )
    with st.spinner("Matching WHO Indicators to MONDO..."):
        selected = who_agent.curate_prop(prop_choice)

if data_choice == "CDC":
    statistic = ["Prevalence", "Rate", "Number"] 
    prop_choice = st.segmented_control(
        "Select statistic:", statistic, default=statistic[0]
    )
    with st.spinner("Matching CDC Indicators to MONDO..."):
        selected = cdc_agent.curate_prop(prop_choice)

if not selected.empty:
    selection_event = st.dataframe(
        selected,
        selection_mode="multi-row",
        on_select="rerun",
        use_container_width=True,
        key="search_table",
        hide_index=True
    )
