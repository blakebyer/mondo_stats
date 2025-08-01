import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
from cdc_query import get_cdc_data
from who_agent import get_mondo_adapter, get_stato_adapter, search_mondo, search_stato
import time

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

mondo = get_mondo_adapter()
stato = get_stato_adapter()

class CDCAnnotation(BaseModel):
    IndicatorName: str
    MONDO_ID: Optional[str] = None
    MONDO_Label: Optional[str] = None
    STATO_ID: Optional[str] = None
    STATO_Label: Optional[str] = None
    Denominator: Optional[int] = None

CDC_AGENT_PROMPT = (
    """
    You are an expert biocurator familiar with the MONDO Disease Ontology and disease terminology. Your task is to help curate ontology terms for given Centers for Disease Control and Prevention proportion estimates.

    You will receive a single input string containing multiple CDC indicator names, separated by newline characters ('\\n'). For example:

    "Short sleep duration among children aged 4 months to 14 years (Crude Prevalence, %)
    \\nBinge drinking prevalence among adults (Crude Prevalence, %)"

    Each line corresponds to a single indicator. For each newline-separated indicator, use your tool-calling ability and knowledge about diseases to extract and return structured information using the CDCAnnotation format with the following fields:

    1. IndicatorName -> Copy the exact value of the indicator as it appears.
                        
    2. MONDO_ID and MONDO_Label -> Use the `search_mondo` tool to identify the disease or condition explicitly mentioned in the indicator.                    

    - First, extract the likely disease portion by removing phrases such as "prevalence among", "Crude Prevalence", "Crude Rate", "prevalence of", "Count of", "Number", or "Rate" to isolate the disease phrase.
    - Allow synonym flexibility: If the initial search does not yield an appropriate match, you may retry using paraphrased forms (e.g., "carcinoma" for "cancer", "oral cavity" -> "oral", "lip" -> "mouth") to ensure better semantic matches.
    - Critically, do not return overly specific, overly general, or unrelated disease entities (e.g., do not return "X-linked intellectual disability-short stature-overweight syndrome" for "overweight"). The match must refer to the same condition or its direct synonym, not a syndrome or phenotype where the condition is only a secondary feature.
    - When multiple candidates are found, ensure that the selected MONDO term is the **best possible match** in specificity and meaning to the disease mentioned in the indicator. Prefer exact matches or primary labels over broader categories or partial overlaps.

    Only return the `id` and `label` exactly as returned by the tool. Do not guess, infer, or invent labels. If no appropriate match is found, leave both fields blank.
                        
    3. STATO_ID and STATO_Label -> Use a single call to `search_stato` to retrieve both the STATO ID and its label from the IndicatorName parentheses, which takes the format "(datavaluetype, datavalueunit)", e.g. "(Crude Prevalence, %)" in IndicatorName = "Binge drinking prevalence among adults (Crude Prevalence, %)". In this example, you would return STATO:0000412 and label "prevalence".

    Examples of STATO_Labels include "prevalence", "rate", "incidence", and "count". 
    One special case is datavaluetype "Number" which should be mapped to STATO:0000047 and label "count" even if the IndicatorName prefix contains the word "incidence".

    Only return the `id` and `label` exactly as returned by the tool. Do not guess, infer, or invent labels. If not found, leave both fields blank. 
                        
    4. Denominator -> Extract the denominator value from the indicator name. For example, if the text says "per 100,000", return 100000. If it says "per 1000", return 1000. If it says "%" return 100. If no denominator is present or is not clearly extractable, leave this field blank.

    IMPORTANT: A tool's response will include both the ID and the label if found, therefore do not invent new term IDs or labels, only use those found in the ontology. You should prioritize speed and accuracy. Return only the JSON list. Do not include explanations, commentary, or free text. Each object must match the CDCAnnotation schema exactly.
    """
)

prop_agent = Agent(
    model="openai:gpt-4o",
    output_type=List[CDCAnnotation],
    system_prompt=CDC_AGENT_PROMPT,
    tools=[search_mondo, search_stato],
)

def curate_prop(proportion: str, batch_size=10, sleep=2):
    """A high level function to curate MONDO and STATO terms from CDC rate, prevalence, or number (count) data"""
    cdc_df = get_cdc_data(proportion)
    cdc_df["IndicatorName"] = (
    cdc_df["question"] + " (" + cdc_df["datavaluetype"] + ", " + cdc_df["datavalueunit"] + ")"
    )
    indicators = cdc_df["IndicatorName"].dropna().unique()
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
    final_df = results_df.merge(cdc_df, on='IndicatorName', how="inner")
    return final_df