import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

from data_process import (DataTransform,
                          compute_metrics,
                          convert_to_bio,
                          convert_bio_tags,
                          target_labels,
                          tag_values)

# bert_name - имя модели или каталог с обученной моделью
# bert_tuned - каталог с обученной моделью (+ токенизатор и конфигурационные файлы)

# были попробованы такие модели:

# bert_name = "Babelscape/wikineural-multilingual-ner"
# bert_name = 'syndi-models/ner-english-fast'
# bert_name = 'swtb/XLM-RoBERTa-Base-Conll2003-English-NER-Finetune'
# bert_name = 'glimmerz/xlmroberta-ner-english'
# bert_name = 'DeepPavlov/rubert-base-cased-conversational'
# bert_name = 'dbmdz/bert-large-cased-finetuned-conll03-english'
# bert_name = 'Jean-Baptiste/roberta-large-ner-english'
# bert_name = 'swtb/XLM-RoBERTa-Base-Conll2003-English-NER-Finetune'
# bert_name = 'yqelz/xml-roberta-large-ner-russian'

# bert_name = 'bert-base-cased'
# bert_name = 'xlm-roberta-large'
# bert_name = r"Z:\python-datasets\LitBank NER 2023\bert-base-cased_7_4"

bert_name = r"Z:\python-datasets\LitBank NER 2023\bert-large-cased_7_4"

bert_tuned = bert_name

dts = DataTransform(model_name=bert_name, model_path=bert_tuned,
                    tokenizer=AutoTokenizer,
                    token_classification=AutoModelForTokenClassification
                    )

if 'PER' in dts.tag2idx:
    convert_function = convert_to_bio
else:
    convert_function = convert_bio_tags

dts.idx2tag = {0: 'O',
               1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'}
print(dts.idx2tag)

dts.tag2idx = dict(sorted(dts.tag2idx.items(), key=lambda x: x[-1]))

# print(dts.tag2idx)
# print(dts.tag2idx.keys())


# train, sentences, labels, row_labels = dts.read_dataset('train_data_sent.csv')

train, sentences, labels, row_labels = dts.read_dataset('test_data_no_labels_sent.csv',
                                                        train_data=False)

sample = len(sentences)

# количество примеров для опытов
# sample = 1

sentences = sentences[:sample]
labels = labels[:sample]

print('tag_values[:-1]', tag_values[:-1], '\n')

true_labels = []
for row_labels in labels:
    true_labels.extend(list(row_labels))

predict_labels = []

for row in tqdm(sentences, total=len(sentences)):
    labels, results = dts.get_entities(row)
    if sample < 10:
        print('pred_labels', labels)
        print('results', results)

    labels = [label.replace('-SM1', 'PER') for label in labels]

    #
    # labels = convert_tags(labels)

    labels = convert_function(labels)

    # оставляем только нужные метки, остальные заменяем на 'O'
    labels = [('O', label)[label in target_labels] for label in labels]

    predict_labels.extend(labels)
    if sample < 10:
        print(len(results), results)

print(len(true_labels), len(predict_labels))
if sample < 10:
    print('true_labels', true_labels)
    print('pred_labels', predict_labels)

# Создаём DataFrame
df = pd.DataFrame({
    'ID': range(len(predict_labels)),
    'tag': predict_labels
})

# Сохраняем DataFrame в файл
df.to_csv('submission.csv', index=False)

compute_metrics(true_labels, predict_labels)
