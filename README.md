# MONDO Disease Prevalence Statistics

This repository hosts code to create disease prevalence summaries for Monarch Disease Ontology (MONDO) disease entities from the most up-to-date World Health
Organization (WHO) and Centers for Disease Control and Prevention (CDC) data.

To use this app, install Python 3.13+. Clone this repository and run on your favorite shell:

    cd src/
    pip install requirements.txt
    streamlit run app.py

Clicking on WHO or CDC buttons automatically generates dotplot summaries of disease prevalence by country. Loading MONDO and STATO as SQLite databases (source ontologies used in this project) may take several minutes. 
