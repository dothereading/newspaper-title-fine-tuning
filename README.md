# newspaper-title-fine-tuning
Using titles of newspaper articles since 1989 from countries all over the world to predict conflict by country with sentence transformers.
Will need df_merged.csv (produced from conflict forecast pipeline), and df_corpus.csv, an internal document of newspaper articles from all over the world.
 
 * `01_titlecleaner.py` : uses the spaCy library to clean the titles, removing country names, people, etc. Produces `titlescleaned.csv` and `titles_cleanedcombined.csv`.
 * `02_finetune.py` : fine tunes a sentence transformer model to make predictions on hard onset. Model is automatically saved.
 * `03_predict.py` : load our model and apply it to the entire corpus of documents.

`df_merged.csv` can now be put into the pipeline with two new columns: 'preds_36_titles' and 'probas_36_titles'.