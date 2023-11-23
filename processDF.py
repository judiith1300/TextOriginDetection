from datasets import load_dataset
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BartTokenizer, BartForConditionalGeneration


def Summarize(text):
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    segment_length = 1024
    segments = [text[i:i + segment_length] for i in range(0, len(text), segment_length)]

    summarized_text = ""
    for segment in segments:
        inputs = tokenizer(segment, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs.input_ids, max_length=1024, do_sample=True)
        summarized_text += tokenizer.decode(summary_ids[0], skip_special_tokens=True) + " "

    return summarized_text
def paraphrase_text(text):

    model_name_paraphrase = 'facebook/bart-large-cnn'
    model_paraphrase = BartForConditionalGeneration.from_pretrained(model_name_paraphrase)
    tokenizer_paraphrase = BartTokenizer.from_pretrained(model_name_paraphrase)

    segment_length = 1024
    segments = [text[i:i + segment_length] for i in range(0, len(text), segment_length)]

    paraphrased_text = ""
    for segment in segments:
        inputs = tokenizer_paraphrase(segment, return_tensors="pt", max_length=1024, truncation=True)
        paraphrased_ids = model_paraphrase.generate(inputs.input_ids, max_length=1024, do_sample=True)
        paraphrased_text += tokenizer_paraphrase.decode(paraphrased_ids[0], skip_special_tokens=True) + " "
    return paraphrased_text

dataset = load_dataset("symanto/autextification2023", 'detection_en')
all_text_list = []
for split in dataset.keys():
    split_text_list = dataset[split]['text']
    all_text_list.extend(split_text_list)
all_text_list = all_text_list[:100]
all_text = ' '.join(all_text_list)
summarization = Summarize(all_text)
paraphrasing = paraphrase_text(summarization)

output_data = {
    "original_text": all_text,
    "summary": summarization,
    "paraphrased_text": paraphrasing
}

output_file = 'output_data.json'
with open(output_file, 'w') as json_file:
    json.dump(output_data, json_file, indent=4)

print(f"Datos guardados exitosamente en '{output_file}'")

