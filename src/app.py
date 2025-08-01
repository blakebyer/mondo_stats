import streamlit as st
import cdc_agent
import who_agent
import pandas as pd
import plotly.express as px
import numpy as np

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
    
    selection_event = st.dataframe(
        selected,
        selection_mode="multi-row",
        on_select="rerun",
        use_container_width=True,
        key="search_table",
        hide_index=True
    )
    plot_df = selected[['IndicatorName', 'MONDO_ID', 'MONDO_Label', 'STATO_ID', 'STATO_Label', 'Denominator', 'Logic', 'SpatialDimType', 'SpatialDim', 'TimeDimType', 'TimeDimensionValue', 'NumericValue', 'Low', 'High']].dropna().drop_duplicates()
    plot_df["NormalizedValue"] = np.divide(
        plot_df["NumericValue"],
        plot_df["Denominator"],
        out=np.full_like(plot_df["NumericValue"], np.nan, dtype=np.float64),
        where=(plot_df["Denominator"] != 0) & plot_df["Denominator"].notna() & plot_df["NumericValue"].notna()
    )

    spatial_dims = plot_df['SpatialDim'].dropna().unique()
    selected_dims = st.multiselect(
        "Select countries:",
        options=sorted(spatial_dims),
        default=sorted(spatial_dims)[:6]
    )
    filtered_df = plot_df[plot_df["SpatialDim"].isin(selected_dims)]

    fig = px.scatter(
        filtered_df, x="TimeDimensionValue", y="NormalizedValue", color='MONDO_Label', facet_row="SpatialDim",
        labels={
        "TimeDimensionValue": "Year",
        "NormalizedValue": "Value",
        "MONDO_Label": "MONDO Label",
        "SpatialDim": "Country"
        },
        title="Trends by Country"
    )
    fig.update_layout(
    font=dict(size=20),
    title=dict(font=dict(size=24)),  # override for title
    legend=dict(font=dict(size=20)))
    st.plotly_chart(fig)


if data_choice == "CDC":
    statistic = ["Prevalence", "Rate", "Number"] 
    prop_choice = st.segmented_control(
        "Select statistic:", statistic, default=statistic[0]
    )
    with st.spinner("Matching CDC Indicators to MONDO..."):
        selected = cdc_agent.curate_prop(prop_choice)
    
    selection_event = st.dataframe(
        selected,
        selection_mode="multi-row",
        on_select="rerun",
        use_container_width=True,
        key="search_table",
        hide_index=True
    )
    plot_df = selected[['IndicatorName', 'MONDO_ID', 'MONDO_Label', 'STATO_ID', 'STATO_Label', 'Denominator', 'yearstart', 'yearend', 'locationabbr','locationdesc', 'datavalue', 'lowconfidencelimit', 'highconfidencelimit', 'stratificationcategory1', 'stratification1']].dropna().drop_duplicates()

    # plot_df["NormalizedValue"] = np.divide(
    #     plot_df["datavalue"],
    #     plot_df["Denominator"],
    #     out=np.full_like(plot_df["datavalue"], np.nan, dtype=np.float64),
    #     where=(plot_df["Denominator"] != 0) & plot_df["Denominator"].notna() & plot_df["datavalue"].notna()
    # )

    spatial_dims = plot_df['locationdesc'].dropna().unique()
    selected_dims = st.multiselect(
        "Select states:",
        options=sorted(spatial_dims),
        default=sorted(spatial_dims)[:6]
    )
    filtered_df = plot_df[plot_df["locationdesc"].isin(selected_dims)]

    fig = px.scatter(
        filtered_df, x="yearstart", y="datavalue", color='MONDO_Label', facet_row="locationdesc",
        labels={
        "yearstart": "Year",
        "datavalue": "Value",
        "MONDO_Label": "MONDO Label",
        "locationdesc": "State"
        },
        title="Trends by State"
    )
    fig.update_layout(
    font=dict(size=20),  # sets default font size for most elements
    title=dict(font=dict(size=24)),  # override for title
    legend=dict(font=dict(size=20)))
    st.plotly_chart(fig)
    
