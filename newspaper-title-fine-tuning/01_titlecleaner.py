import pandas as pd
import spacy
import subprocess
import sys

# Check if the spaCy model is already installed, and if not, install it
try:
    import en_core_web_sm
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
        )
nlp = spacy.load("en_core_web_sm")

# ---------------
# -- LOAD DATA --
# ---------------

# Set paths to datasets
isocode_path = ("/Users/Miguel/Documents/GitHub/s05_coup/input/"
                "country_isocode.csv")
corpus_path = "data/df_corpus.csv"

# Load Corpus
chunk_size = 10000  # Adjust based on your memory constraints
columns_needed = ["country", "date", "title"]
selected_data = pd.DataFrame()

# Read the CSV file in chunks
for chunk in pd.read_csv(corpus_path, chunksize=chunk_size,
                         usecols=columns_needed):
    selected_data = pd.concat([selected_data, chunk])
print("Corpus data loaded.")

# Use isocodes instead of full country names
code_conv = pd.read_csv(isocode_path)
selected_data = pd.merge(selected_data, code_conv, on="country", how="left")

# Create year-month date format called 'period'
selected_data["date"] = selected_data["date"].astype(str)
selected_data["period"] = selected_data["date"].str[:6]
selected_data["period"] = selected_data["period"].astype(int)

# Ensure titles are strings
selected_data["title"] = selected_data["title"].astype(str)
print("df_corpus prepped. This next part will take many hours.")

# ---------------
# --- MASKING ---
# ---------------

# Function to mask specific terms from newspaper titles
def mask_entities(text: str) -> str:
    """
    This function masks geopolitical entities, locations, organizations,
    national/religious/political groups, and people.
    """
    doc = nlp(text)
    for ent in doc.ents:
        # Geopolitical entity (often includes both cities and countries)
        if (ent.label_ == "GPE"):
            text = text.replace(ent.text, "#GPE#")
        # Locational entity (more general, can include regions, water etc.)
        elif (ent.label_ == "LOC"):
            text = text.replace(ent.text, "#LOC#")
        # Organizations (companies etc.)
        elif (ent.label_ == "ORG"):
            text = text.replace(ent.text, "#ORG#")
        # National, religious or political groups.
        elif ent.label_ == "NORP":
            text = text.replace(ent.text, "#NORP#")
        # Person
        elif ent.label_ == "PERSON":
            text = text.replace(ent.text, "#PER#")
    return text

# Apply function to titles in corpus and save to csv
selected_data["cleaned_title"] = [
    mask_entities(text) for text in selected_data["title"]
]
cleaned_df = selected_data[["isocode", "period", "cleaned_title", "title"]]
cleaned_df.to_csv("data/titles_cleaned.csv", index=False)
print("Entities masked.")

# ---------------
# -- COMBINING --
# ---------------

# make a df with a list of titles for every country for every period
combined_titles = (
    cleaned_df.groupby(["isocode", "period"])["cleaned_title"]
    .apply(". ".join)
    .reset_index()
)
combined_titles.to_csv("data/titles_cleanedcombined.csv", index=False)
print("Csv ouputted.")
