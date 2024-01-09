import pandas as pd
import subprocess
import sys

def install_package(package_name):
    try:
        __import__(package_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name]
            )

install_package("setfit")
from setfit import SetFitModel

# -----------------------
# -- FILEPATHS + NAMES --
# -----------------------

model_name = "best_model_20240108-192025"  ## update ##
last_monthdate = 202310  ## update ##

path_repo = path_repo = "/content/drive/MyDrive/FCDO/"
model_path = path_repo + "models/" + model_name
path_merged = path_repo + "data/df_merged.csv"
path_titles = path_repo + "data/titles_cleanedcombined.csv"

# -----------------------
# -- LOAD DATA + MODEL --
# -----------------------

df_mer = pd.read_csv(path_merged)
try:
    df_mer = df_mer.drop(
        columns=["preds_36_titles", "probas_36_titles"],
        axis=1
    )
except:
    pass
df_titles = pd.read_csv(path_titles)
recent_df = df_mer[df_mer["updated"] == last_monthdate]

# merge titles with labels ('ons_armedconf_12', etc.)
df = pd.merge(
    df_titles[["isocode", "period", "cleaned_title"]],
    recent_df[
        [
            "isocode",
            "period",
            "ons_armedconf_12",
            "ons_armedconf_36",
            "ons_armedconf_60",
        ]
    ],
    how="left",
    on=["isocode", "period"],
)
df = df.drop_duplicates(subset=["isocode", "period"])
print("Data loaded.")

model = SetFitModel._from_pretrained(model_path)
print("Model loaded.")

# -----------------
# -- APPLY MODEL --
# -----------------

titles = [str(text) for text in df["cleaned_title"]]

preds_out = model.predict(titles)
probas_out = model.predict_proba(titles)
preds_out = preds_out.tolist()
probas_out = probas_out.numpy()[:, 1].tolist()
print("Predictions made.")

preds_out_int = [int(pred) for pred in preds_out]
probas_out_round = [round(proba, 3) for proba in probas_out]

df["preds_36_titles"] = preds_out_int
df["probas_36_titles"] = probas_out_round

df_out = pd.merge(
    df_mer,
    df[["isocode", "period", "preds_36_titles", "probas_36_titles"]],
    on=["isocode", "period"],
    how="left",
)

# Forward fill but backward fill in case first val is blank
df_out["preds_36_titles"] = (
    df_out["preds_36_titles"].fillna(method="ffill").fillna(method="bfill")
)
df_out["probas_36_titles"] = (
    df_out["probas_36_titles"].fillna(method="ffill").fillna(method="bfill")
)
print("Na's filled.")

df_out.to_csv(path_merged, index=False)
print("df_merged updated in: ", path_merged)
