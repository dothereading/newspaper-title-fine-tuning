import datetime
import numpy as np
import os
import pandas as pd
import random
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import subprocess
import sys

def install_package(package_name):
    try:
        __import__(package_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name]
            )

# Ensure 'setfit' and 'datasets' are installed
install_package("setfit")
install_package("datasets")

from datasets import load_dataset, Dataset, concatenate_datasets
from setfit import SetFitModel
from setfit import Trainer, TrainingArguments


# -----------------------
# -- FILEPATHS + NAMES --
# -----------------------

last_monthdate = 202310  ## update ##
path_repo = "/content/drive/MyDrive/FCDO/"
path_merged = path_repo + "data/df_merged.csv"
path_titles = path_repo + "data/titles_cleaned.csv"
# here we load all titles before combining ^

# ---------------
# -- LOAD DATA --
# ---------------

df_mer = pd.read_csv(path_merged)
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

# -----------------------------
# -- CREATE TRAIN + TEST SET --
# -----------------------------

# training examples (anything before 201701)
pos_36_tr = df[(df["ons_armedconf_36"] == 1) & (df["period"] < 201701)]
neg_36_tr = df[(df["ons_armedconf_36"] == 0) & (df["period"] < 201701)]

# test examples
pos_36_te = df[(df["ons_armedconf_36"] == 1) & (df["period"] >= 201701)]
neg_36_te = df[(df["ons_armedconf_36"] == 0) & (df["period"] >= 201701)]

positive_examples = list(pos_36_tr["cleaned_title"])
positive_examples_te = list(pos_36_te["cleaned_title"])
negative_examples = list(neg_36_tr["cleaned_title"])
negative_examples_te = list(neg_36_te["cleaned_title"])

# From all data select n positive examples and m negative examples
n = 50
m = 2 * n
print(
    f"We'll use {n} positive examples and",
    "{m} negative examples to train model."
)
positive_examples = random.sample(positive_examples, n)
negative_examples = set(negative_examples)  # removes any potential duplicates.
negative_examples = random.sample(negative_examples, m)

# Here we prepare our data in the format for Hugging Face models
texts_train = negative_examples + positive_examples
texts_train = [str(text) for text in texts_train]
labels = [0] * m + [1] * n
labels_text = ["no_conflict_predicted"] * m + ["conflict_predicted"] * n
texts_train, labels, labels_text = shuffle(texts_train, labels, labels_text)
dataset = Dataset.from_dict(
    {"text": texts_train, "label": labels, "label_text": labels_text}
)

# Prepare our test dataset
texts_te = negative_examples_te + positive_examples_te
texts_te = [str(text) for text in texts_te]
n_2 = len(positive_examples_te)
m_2 = len(negative_examples_te)
labels_te = [1] * n_2 + [0] * m_2
labels_text_te = ["conflict_predicted"] * n_2 + ["no_conflict_predicted"] * m_2
texts_te, labels_te, labels_text_te = shuffle(
    texts_te,
    labels_te,
    labels_text_te
    )
dataset_te = Dataset.from_dict(
    {"text": texts_te, "label": labels_te, "label_text": labels_text_te}
)

print("Datasets prepped.")

# -----------------
# -- TRAIN MODEL --
# -----------------

# Instantiate your custom model
model = SetFitModel.from_pretrained(
    "sentence-transformers/distiluse-base-multilingual-cased-v1"
)

# Define training arguments
args = TrainingArguments(
    body_learning_rate=1e-4,
    head_learning_rate=1e-4,
    batch_size=32,
    num_epochs=1,
)

# Create trainer with new Trainer class and TrainingArguments
trainer = Trainer(
    model=model, args=args, train_dataset=dataset, eval_dataset=dataset_te
)

trainer.train()

print("Model trained.")

# ----------------
# -- SAVE MODEL --
# ----------------

# Define the path where you want to create the directory
dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_path = path_repo + f"models/best_model_{dt}"
os.makedirs(model_path, exist_ok=True)

model._save_pretrained(model_path)
print("Model saved in: ", model_path)

# ----------------
# -- EVAL MODEL --
# ----------------

# Run predictions and probabilities on test datset
preds = model.predict(dataset_te["text"])
probas = model.predict_proba(dataset_te["text"])
preds = preds.tolist()
probas = probas.numpy()[:, 1].tolist()

# Calculate error metrics including AUC on test set
auc_sc = roc_auc_score(dataset_te["label"], preds)
print("AUC:", auc_sc)
accuracy = accuracy_score(dataset_te["label"], preds)
print("Accuracy:", accuracy)
cm = confusion_matrix(dataset_te["label"], preds)
print("Confusion Matrix:")
print(cm)

# Extract texts and probabilities
dataset_te = dataset_te.add_column("preds", preds)
dataset_te = dataset_te.add_column("probas", probas)
texts1 = [item["text"] for item in dataset_te]
probas1 = np.array([item["probas"] for item in dataset_te])
labels1 = np.array([item["label"] for item in dataset_te])
preds_range = {
    i: [
        text
        for text, prob, label in zip(texts1, probas1, labels1)
        if (prob < i / 100) & (prob > (i - 10) / 100)
    ]
    for i in range(10, 110, 10)
}

# Create a sample for every 10%, and save to .txt file
path_sample = path_repo + f"models/best_model_{dt}/0_testset_sample.txt"
with open(path_sample, "w") as file:
    file.write(f"Testset AUC: {auc_sc}\n" +
               f"Testset accuracy: {accuracy}\n" +
               f"CM: {cm}\n\n")
    for key, value in preds_range.items():
        file.write(f"Range: [{key-10};{key}]: " +
                   f"{len(value)/len(texts1) * 100:.2f} % of lines\n\n")
        # Shuffle and take a sample of 10 texts and their corresponding labels
        sample = np.random.permutation(
            [(text, label) for text, label in zip(
                texts1,
                labels1
                ) if text in value]
        )[:10]
        for text, label in sample:
            file.write(f"Label: {label}\nText: {text}\n---\n")
        file.write("---\n\n")


print("A sample of classifications are found in:", path_sample)
