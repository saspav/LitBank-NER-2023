import os
import re
import numpy as np
import pandas as pd
import subprocess

from glob import glob
from tqdm import tqdm
from collections import defaultdict, Counter
from data_process import (DataTransform,
                          convert_bio_tags,
                          name_tokens,
                          make_one_tag,
                          merge_tags
                          )

# пытаемся импортировать самодельный экспорт в эксель с красивостями
try:
    from df_addons import df_to_excel
except ModuleNotFoundError:
    df_to_excel = lambda sdf, spt, *args, **kwargs: sdf.to_excel(spt, index=False)

__import__('warnings').filterwarnings("ignore")

tqdm.pandas()

dts = DataTransform(load_model=False)

df, *_ = dts.read_dataset('test_data_no_labels_sent.csv', train_data=False)

df.insert(2, 'prev_token', df['token'].shift(1))

print(df)

files = glob(r'.\compare\*.csv')
merge_cols = []
for file in files:
    print(f'\n{file}')
    tmp = pd.read_csv(file)
    name_file = file.split('\\')[-1].split('.')[0]
    tmp.columns = ['ID', name_file]
    merge_cols.append(name_file)

    df = df.merge(tmp, on='ID')

    df[name_file] = df.progress_apply(lambda row: make_one_tag(row, name_file), axis=1)

df['not_O'] = df.apply(lambda row: any(row[col] != 'O' for col in merge_cols), axis=1)

print('\nПоиск самого частотного тега для токена')
df['tag'] = df.progress_apply(lambda row: merge_tags(row[merge_cols].to_list()), axis=1)

tokens = df['token'].str.replace('-u-ell', 'uel').to_list()
tags = df['tag'].to_list()

token_tag = []
token_outs = ('American', 'Jack')
lower_words = set()
for token, tag in zip(tokens, tags):
    if token.isalpha() and token not in token_outs:
        if token[0].isupper() and len(token) >= 2:
            token_tag.append((token, tag))
        elif token.islower():
            lower_words.add(token)

print('Составляем список тегов для слов с заглавной буквы')
labels = defaultdict(list)
for token, tag in token_tag:
    if tag != 'O' and token.lower() not in lower_words:
        labels[token].append(tag.replace('I-', 'B-'))

print('len(token_tag)', len(token_tag))
print('len(labels)', len(labels))

tlb = pd.DataFrame(data=labels.items(), columns=['token', 'tags'])
col_width = enumerate([20, 64])
df_to_excel(tlb, 'token_tags.xlsx', ins_col_width=col_width)

print("labels['Morel']", labels['Morel'])

print('Поиск самого частотного тега для Имен')
common_labels = dict()
for key, tag_values in labels.items():
    if tag_values:
        common_labels[key] = Counter(sorted(tag_values)).most_common()[0][0]

# Обрабатываем только "X-PER"
print('Замена тегов для Имен на самый частотный')
for idx, token in enumerate(tokens.copy()):
    if (common_labels.get(token, '').endswith('PER') and not tags[idx].endswith('PER') and
            token in name_tokens and token[0].isupper() and token in common_labels):
        tags[idx] = common_labels[token]


df['tag'] = convert_bio_tags(tags)

print('Сохранение результатов...')
df_to_excel(df, 'merged_tag.xlsx')

col_width = enumerate([7, 18, 20, 13] + [13] * (len(merge_cols) + 2))
df_to_excel(df, 'merged.xlsx', ins_col_width=col_width)

# Сохраняем DataFrame в файл
name_submit = f"merged_{'_'.join(merge_cols)}.csv"
df[['ID', 'tag']].to_csv(name_submit, index=False)

# Установим переменную окружения для Kaggle
os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.path.expanduser('~'), '.kaggle')

# Используем API Kaggle для отправки файла
command = ['kaggle', 'competitions', 'submit', '-c', 'litbank-ner-2023', '-f',
           name_submit, '-m', f'merged_{len(merge_cols)}_models']

# Выполняем команду
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Выводим результат
print(result.stdout)
print(result.stderr)
