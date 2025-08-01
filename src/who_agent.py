import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic import BaseModel
from typing import Optional, List
from oaklib import get_adapter
from oaklib.datamodels.search import SearchConfiguration
import pandas as pd
import threading
from who_query import get_who_data
import time

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

mondo = get_adapter("sqlite:obo:mondo")
stato = get_adapter("sqlite:obo:stato")

mondo_lock = threading.Lock()
stato_lock = threading.Lock()

HUMAN_DISEASE_ROOT = "MONDO:0700096"

def is_human_disease(curie: str) -> bool:
    ancestors = set(mondo.ancestors(curie))
    return HUMAN_DISEASE_ROOT in ancestors

def search_mondo(label: str) -> List[dict]:
    """Search the MONDO Ontology for disease identifiers."""
    with mondo_lock:
        results = list(mondo.basic_search(label, SearchConfiguration(is_partial=True)))
        data = []
        for curie in results:
            if not is_human_disease(curie):
                continue
            data.append({
                "id" : curie,
                "label" : mondo.label(curie),
                "definition": mondo.definition(curie),
            })
        return data

def search_stato(label: str) -> List[dict]:
    """Search the STATO Ontology for Prevalence, Incidence, or Count identifiers."""
    with stato_lock:
        results = list(stato.basic_search(label))
        data = []
        for curie in results:
            data.append({
                "id" : curie,
                "label" : stato.label(curie),
                "definition": stato.definition(curie),
            })
        return data

# Data schema
class WHOAnnotation(BaseModel):
    IndicatorName: str
    MONDO_ID: Optional[str] = None
    MONDO_Label: Optional[str] = None
    STATO_ID: Optional[str] = None
    STATO_Label: Optional[str] = None
    Denominator: Optional[int] = None

PROP_AGENT_PROMPT = ("""
You are an expert biocurator familiar with the MONDO Disease Ontology and disease terminology. Your task is to help curate ontology terms for given World Health Organization proportion estimates.

You will receive a single input string containing multiple WHO indicator names, separated by newline characters ('\\n'). For example:

"Malaria incidence per 100 000 population\\nAlcoholic psychosis, incidence, per 100,000"

Each line corresponds to a single indicator. For each newline-separated indicator, use your tool-calling ability and knowledge about diseases to extract and return structured information using the WHOAnnotation format with the following fields:

1. IndicatorName -> Copy the exact value of the indicator as it appears.
                     
2. MONDO_ID and MONDO_Label -> Use the `search_mondo` tool to identify the disease or condition explicitly mentioned in the indicator. Return only exact matches or direct synonyms of the disease entity described. If the initial search does not yield an appropriate match, you may retry with common synonyms, abbreviations, or alternative phrasings (e.g., "carcinoma" instead of "cancer"). 

Critically, do not return overly specific or unrelated disease entities (e.g., do not return "X-linked intellectual disability-short stature-overweight syndrome" for "overweight"). The match must refer to the same condition or its direct synonym, not a syndrome or phenotype where the condition is only a secondary feature.

Only return the `id` and `label` exactly as returned by the tool. Do not guess, infer, or invent labels. If no appropriate match is found, leave both fields blank.
                     
3. STATO_ID and STATO_Label -> Use a single call to `search_stato` to retrieve both the STATO ID and its label (e.g. for "prevalence", "incidence", or "count"). If not found, leave both fields blank. For instance, prevalence is STATO:0000412 and the label is "prevalence".
                     
4. Denominator -> Extract the denominator value from the indicator name. For example, if the text says "per 100,000", return 100000. If it says "per 1000", return 1000. If it says "%" return 100. If no denominator is present or is not clearly extractable, leave this field blank.

IMPORTANT: A tool's response will include both the ID and the label if found, therefore not invent new term IDs or labels, only use those found in the ontology. You should prioritize speed and accuracy. Return only the JSON list. Do not include explanations, commentary, or free text. Each object must match the WHOAnnotation schema exactly.
""")

prop_agent = Agent(
    model="openai:gpt-4o",
    output_type=List[WHOAnnotation],
    system_prompt=PROP_AGENT_PROMPT,
    tools=[search_mondo, search_stato],
)

def curate_prop(proportion="incidence", batch_size=10, sleep=2):
    """A high level function to curate MONDO and STATO terms from WHO incidence, prevalence, or count data"""
    who_df = get_who_data(proportion)
    indicators = who_df["IndicatorName"].dropna().unique()
    results = []

    for i in range(0, len(indicators), batch_size):
        chunk = indicators[i:i + batch_size]
        batched_input = "\n".join(chunk)

        try:
            result = prop_agent.run_sync(batched_input)

            if isinstance(result.output, list):
                results.extend(result.output)
            else:
                results.append(result.output)

        except Exception as e:
            print(f"Batch {i // batch_size} failed: {e}")
            continue
    
        time.sleep(sleep)   

    results_dicts = [r.model_dump() for r in results]
    results_df = pd.DataFrame(results_dicts)
    final_df = results_df.merge(who_df, on='IndicatorName', how="inner")
    return final_df
