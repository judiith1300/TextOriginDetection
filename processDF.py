from datasets import load_dataset
import pandas as pd
import random
from transformers import pipeline,  T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 300)

def summarize_with_t5(text):
    model_name = 't5-large'
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    preprocess_text = "summarize: " + text
    tokenized_text = tokenizer.encode(preprocess_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(tokenized_text, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def paraphrase_text(text):
    model_name_paraphrase = 'facebook/bart-large-cnn'
    model_paraphrase = BartForConditionalGeneration.from_pretrained(model_name_paraphrase)
    tokenizer_paraphrase = BartTokenizer.from_pretrained(model_name_paraphrase)

    inputs = tokenizer_paraphrase([text], return_tensors='pt', max_length=1024, truncation=True)
    paraphrased_ids = model_paraphrase.generate(**inputs)
    paraphrased_text = tokenizer_paraphrase.decode(paraphrased_ids[0], skip_special_tokens=True)
    return paraphrased_text

# Load the dataset and select a random sample of 100 rows
dataset = load_dataset("symanto/autextification2023", 'detection_en')
df_train = dataset['train'].to_pandas()
df_test = dataset['test'].to_pandas()
df = pd.concat([df_train, df_test], ignore_index=True).sample(n=5)

df['summary_text'] = df['text'].apply(summarize_with_t5)
#df['models_applied'] = 't5-large'
df['paraphrased_summary'] = df['summary_text'].apply(paraphrase_text)

df.to_csv('test.csv')
