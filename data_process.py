import re
import numpy as np
import pandas as pd
import torch
import string

from tqdm import tqdm
from collections import Counter, defaultdict
from transformers import BertTokenizer, BertForTokenClassification, AutoConfig
from transformers import DebertaV2Tokenizer, DebertaV2ForTokenClassification
from transformers import RobertaTokenizer, RobertaForTokenClassification
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics import f1_score, classification_report

# пытаемся импортировать самодельный экспорт в эксель с красивостями
try:
    from df_addons import df_to_excel
except ModuleNotFoundError:
    df_to_excel = lambda sdf, spt, *args, **kwargs: sdf.to_excel(spt, index=False)

__import__('warnings').filterwarnings("ignore")

MAX_LEN = 512
OVERLAP = 0.2

target_labels = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

tag_values = ['O'] + target_labels + ['PAD']
tag2idx = {tag: idx for idx, tag in enumerate(tag_values)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}


def compute_metrics(true_labels, pred_labels):
    """
    Подсчет метрик
    :param true_labels: список списков с истинными метками для каждого слова
    :param pred_labels: список списков с предсказанными метками для каждого слова
    :return: f1_score
    """
    # true_tags = sum(true_labels, [])
    # pred_tags = sum(pred_labels, [])

    true_tags = true_labels
    pred_tags = pred_labels

    score = f1_score(true_tags, pred_tags, labels=tag_values[:-1], average='macro')
    print(f'f1_score macro = {score}\n')

    # score = f1_score(true_tags, pred_tags, labels=labels, average='weighted')
    # print(f'f1_score weighted = {score}\n')

    # Вычисление F1-меры для каждого класса
    f1_scores = f1_score(true_tags, pred_tags, labels=tag_values[:-1], average=None)
    print(f'f1_score = {f1_scores}\n')

    print(classification_report(true_tags, pred_tags, zero_division=1))

    return score


class DataTransform:
    """ Класс для поиска сущностей"""

    def __init__(self, model_name=None, model_path=None,
                 tokenizer=None, token_classification=None,
                 load_model=True, cuda=True):
        """
        Инициализация экземпляра класса
        :param model_name: имя модели
        :param model_path: путь к предобученной модели
        :param tokenizer: токенизатор
        :param token_classification: классификатор
        :param cuda: использовать GPU
        """

        # Используем GPU если доступно
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')

        # путь к локальной модели
        if model_name is None:
            self.model_name = './model'
        else:
            self.model_name = model_name

        # путь к локальной модели
        if model_path is None:
            self.model_path = './model'
        else:
            self.model_path = model_path

        self.id2label = None
        self.label2id = None

        if load_model:

            # Загружаем конфигурацию модели
            config = AutoConfig.from_pretrained(self.model_name)

            # Извлекаем словари id2label и label2id
            label2id = config.label2id  # {'O': 0, 'B-PER': 1, 'I-PER': 2, ...}
            id2label = config.id2label  # {0: 'O', 1: 'B-PER', 2: 'I-PER', ...}

            self.tag2idx = label2id
            self.idx2tag = id2label

            if tokenizer is None:
                tokenizer = BertTokenizer

            self.tokenizer = tokenizer.from_pretrained(self.model_name,
                                                       do_lower_case=False)

            if token_classification is None:
                token_classification = BertForTokenClassification
            # Загрузка модели
            self.model = token_classification.from_pretrained(self.model_path,
                                                              num_labels=len(self.tag2idx),
                                                              output_attentions=False,
                                                              output_hidden_states=False,
                                                              ignore_mismatched_sizes=True,
                                                              # from_tf=True,
                                                              )
            self.model.to(self.device)
            self.model.eval()

        else:
            self.model = None

    def tokenize_and_preserve_labels(self, sentence, text_labels):
        tokenized_sentence = []
        labels = []

        for word, label in zip(sentence, text_labels):
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_sentence.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)

        return tokenized_sentence, labels

    def preprocess_sentence(self, words):
        """
        Преобразование текста в нужный формат: токенизация и преобразование входных данных
        :param text: текст
        :return: words, tokenized_words, input_ids, attention_mask
        """
        tokenized_words = []
        for word in words:
            t_word = self.tokenizer.tokenize(word)
            t_word = [w if not i or w[:2] == '##' else f'##{w}' for i, w in enumerate(t_word)]
            tokenized_words.extend(t_word)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_words)

        input_ids = input_ids[:MAX_LEN]

        attention_mask = [1] * len(input_ids)

        return words, tokenized_words, input_ids, attention_mask

    def split_text_with_overlap(self, words, max_len=MAX_LEN, overlap=OVERLAP):
        """
        Разделить текст на части с перекрытием.
        :param words: Предложение из слов для разделения
        :param max_len: максимальная длина части
        :param overlap: процент перекрытия
        :return: список частей текста
        """

        chunk_size = max(max_len - 12, 10)  # Оставляем место для специальных токенов
        overlap_size = int(chunk_size * overlap)
        chunks = []
        start_index = end_index = 0
        while end_index < len(words):
            end_index = start_index + chunk_size
            chunk_words = words[start_index:end_index]
            chunk_text = ' '.join(chunk_words)

            # Токенизируем текст, чтобы убедиться, что кол-во токенов не превышает max_len
            tokens = self.tokenizer.tokenize(chunk_text)
            while len(tokens) > chunk_size and len(chunk_words):
                chunk_words = chunk_words[:-1]
                chunk_text = ' '.join(chunk_words)
                tokens = self.tokenizer.tokenize(chunk_text)

            chunks.append((chunk_words, start_index))

            # У нас один чанк и нечего дальше крутить цикл - виснет
            if len(chunk_words) == len(words):
                break

            # найдем сколько слов входит в перекрытие для вычисления индекса смещения
            tokens = []
            overlap_index = 0
            reversed_words = chunk_words[::-1]
            # пока длина токенов перекрытия меньше размера перекрытия добавляем по слову
            while len(tokens) < overlap_size:
                overlap_index += 1
                tokens = self.tokenizer.tokenize(' '.join(reversed_words[:overlap_index]))

            end_index = start_index + len(chunk_words)
            # Следующая часть начинается с учетом перекрытия
            start_index += len(chunk_words) - overlap_index

        return chunks

    @staticmethod
    def get_words_positions(input_words, pattern):
        found_index = []
        for idx, word in enumerate(input_words):
            if pattern.match(word):
                found_index.append(idx)
        return found_index

    def get_entities(self, words):
        """
        Функция принимает на вход слова и возвращает найденные сущности и их индексы.
        :param words: Предложение из слов
        :return: найденные сущности и их индексы
        """
        # Разделить текст на части с перекрытием
        chunks = self.split_text_with_overlap(words, MAX_LEN, OVERLAP)
        all_labels_positions = []
        all_results = []

        # print(f'\nlen(chunks): {len(chunks)}', 'chunks:', *chunks, sep='\n')

        for chunk, start_index in chunks:
            words, tokenized_words, input_ids, attention_mask = self.preprocess_sentence(
                chunk)

            # Создайте тензоры для входных данных
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(self.device)

            # Прогоните текст через модель для предсказания меток
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)

            logits = outputs[0].detach().cpu().numpy()
            predicted_labels = np.argmax(logits, axis=2)[0]

            labels_positions, result = self.get_entities_with_labels(tokenized_words,
                                                                     predicted_labels,
                                                                     start_index)

            # print('labels_positions:', labels_positions, result)

            all_labels_positions.append(labels_positions)
            all_results.extend(result)

        try:
            # Удаление дублированных меток и приведение к исходному тексту
            labels_positions, final_results = self.merge_chunks_results(all_labels_positions,
                                                                        all_results,
                                                                        len(words))
        except TypeError as err:
            print(err)
            print(words)
            print(all_labels_positions, all_results, len(words), sep='\n')

        # print('lp_fr:', labels_positions, final_results, len(words), sep='\n')

        return sum(labels_positions, []), final_results

    def get_entities_with_labels(self, tokenized_words, predicted_labels, start_index):
        """
        Объединение токенов в сущности с метками
        :param tokenized_words: токенизированные слова
        :param predicted_labels: предсказанные метки
        :param start_index: начальный индекс чанка
        :return:
        """
        current_word = ""
        current_label = []
        words_with_labels = []

        # print('token_words, pred_labels:', tokenized_words, predicted_labels, sep='\n')

        if isinstance(self.tokenizer, DebertaV2Tokenizer):
            for token, label in zip(tokenized_words, predicted_labels):
                if token.startswith('▁'):
                    if current_word:
                        words_with_labels.append((current_word, current_label))
                    current_word = token[1:]
                    current_label = [label]
                else:
                    current_word += token
                    current_label.append(label)

        else:
            for token, label in zip(tokenized_words, predicted_labels):
                if token.startswith("##"):
                    current_word += token[2:]
                    current_label.append(label)
                else:
                    if current_word:
                        words_with_labels.append((current_word, current_label))
                    current_word = token
                    current_label = [label]

        if current_word:
            words_with_labels.append((current_word, current_label))

        # print('words_with_labels', words_with_labels)

        result = []
        labels_positions = []
        for idx, (word, labels) in enumerate(words_with_labels):
            label = Counter(labels).most_common(1)[0][0]
            labels_positions.append(self.idx2tag.get(label, 'O'))
            result.append((word, label, start_index + idx))

        return labels_positions, result

    @staticmethod
    def merge_chunks_results(labels_positions, results, original_length):
        """
        Объединение результатов из перекрывающихся частей в один результат.
        :param labels_positions: Список позиций меток из частей
        :param results: список результатов из частей
        :param original_length: длина оригинального текста в словах
        :return: объединенные метки и результаты
        """
        final_labels_positions = labels_positions.copy()
        final_results = results.copy()

        # print('labels_positions:', labels_positions)
        # print('results:', results)

        # for word, label, idx in results:
        #     if final_results[idx] > 2:
        #         final_results[idx] = label

        return final_labels_positions, final_results

    @staticmethod
    def transform_text_labels(text, labels):
        """
        Формирование списка меток для каждого слова в тексте.
        :param text: текст со словами, разделенными пробелами
        :param labels: список со словарями меток и их позициями
        :return: список меток для каждого слова в тексте
        """
        # если метки - это словарь, засунем его в список
        if isinstance(labels, dict):
            labels = [labels]
        len_words = len(text.split())
        idx_labels = ['O'] * len_words
        for label_dict in labels:
            for key, values in label_dict.items():
                for value in values:
                    if value < len_words:
                        idx_labels[value] = key
        return idx_labels

    @staticmethod
    def process_token(word):
        """
        Функция для обработки слова - удаление знаков пунктуации
        :param word:
        :return:
        """
        if any(char.isalpha() for char in word):
            return word.strip(string.punctuation)
        else:
            return word

    @staticmethod
    def make_cat(tags):
        """
        Присвоение категории предложению
        :param tags: список тегов предложения
        :return: метка категории
        """
        per = any('PER' in tag for tag in tags)
        org = any('ORG' in tag for tag in tags)
        loc = any('LOC' in tag for tag in tags)
        return 4 * per + 2 * org + loc

    def read_dataset(self, path_file, train_data=True):
        """
        Чтение датасета
        :param path_file: полное имя файла с путем
        :param train_data: это трейн?
        :return: df, sentence_rows, sentence_tags, sentence_labels
        """
        with open(path_file, encoding='utf-8') as file:
            data = file.read()

        # Переменные для данных и номер предложения
        data_lines = data.strip().split('\n')
        num_sentence = 0
        rows = []

        # Чтение построчно и запись в структуру данных
        for line in data_lines[1:]:  # Пропуск заголовка
            if line.strip() == '':
                num_sentence += 1
            else:
                line = re.sub(r'"+', '"', line)
                line = re.sub(r'-+', '-', line)
                parts = line.split('\t')
                if not train_data:
                    parts.append('')
                rows.append([int(parts[0]), parts[1], parts[2], num_sentence])

        # Создание DataFrame
        columns = ['ID', 'token', 'tag', 'num_sentence']
        df = pd.DataFrame(rows, columns=columns)

        # Применение функции к колонке 'token'
        df['token'] = df['token'].apply(self.process_token)

        sentence_rows = []
        sentence_tags = []
        sentence_labels = []
        for num in sorted(df.num_sentence.unique()):
            temp = df[df.num_sentence == num]
            tokens = temp.token.values
            tags = temp.tag.values
            sentence_rows.append(tokens)
            sentence_tags.append(tags)
            sentence_labels.append(self.make_cat(tags))
        return df, sentence_rows, sentence_tags, sentence_labels


def convert_bio_tags(pred_labels):
    new_labels = []
    prev_label = 'O'

    for label in pred_labels:
        if label == 'O':
            new_labels.append('O')
            prev_label = 'O'
        else:
            entity_type = label.split('-')[-1]
            if prev_label == 'O' or prev_label.split('-')[-1] != entity_type:
                new_labels.append(f'B-{entity_type}')
            else:
                new_labels.append(f'I-{entity_type}')
            prev_label = label

    return new_labels


def convert_to_bio(true_labels):
    new_labels = []
    prev_label = 'O'

    for label in true_labels:
        if label == 'O':
            new_labels.append('O')
            prev_label = 'O'
        else:
            if prev_label == 'O' or prev_label != label:
                new_labels.append(f'B-{label}')
            else:
                new_labels.append(f'I-{label}')
            prev_label = label

    return new_labels


# обращения - за ними всегда идет имя - взято из Титаника
appeals = {'Mr', 'Mrs', 'Miss', 'Missis', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms', 'Major',
           'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess', 'Jonkheer', 'Madam', 'Captain',
           # 'Lord',
           }

name_tokens = ('Denham', 'Edith', 'Emma', 'Herncastle', 'John', 'Margaret', 'Miguel',
               'Newton', 'Powell', 'Winterbourne')


def make_tag(row, merge_values):
    values = list(row[merge_values].values)
    if row['token'].endswith('s') and not row['token'][0].isalpha():
        return 'O'
    elif row['token'] in appeals:
        # если это обращение к персоне - то это всегда 'B-PER'
        return 'B-PER'
    elif row['prev_token'] in appeals and row['token'].isalpha():
        # если предыдущее слово обращение к персоне, текущий токен буквенный - всегда 'I-PER'
        return 'I-PER'
    elif all(value.endswith('LOC') for value in values):
        return values[-1]
    elif values[-1][-3:] in ('ORG', 'PER'):
        return values[-1]
    # elif values[-1].endswith('PER'):
    #     return values[-1]
    # elif values[0].endswith('PER') and not values[-1].endswith('PER'):
    #     return values[0]
    return 'O'


def make_one_tag(row, col):
    if row['token'].endswith('s') and not row['token'][0].isalpha():
        return 'O'
    elif row['token'] in appeals:
        # если это обращение к персоне - то это всегда 'B-PER'
        return 'B-PER'
    elif row['prev_token'] in appeals and row['token'].isalpha():
        # если предыдущее слово обращение к персоне, текущий токен буквенный - всегда 'I-PER'
        return 'I-PER'
    return row[col]


def merge_tags(tags):
    tags = [tag.replace('I-', 'B-') for tag in tags if tag != 'O']
    if len(tags) >= 3:
        return Counter(sorted(tags)).most_common()[0][0]
    return 'O'


if __name__ == "__main__":
    pass

    # print(tag2idx)
    # print(idx2tag)

    bert_name = "Babelscape/wikineural-multilingual-ner"

    # bert_name = 'syndi-models/ner-english-fast'
    # bert_name = 'swtb/XLM-RoBERTa-Base-Conll2003-English-NER-Finetune'
    # bert_name = 'glimmerz/xlmroberta-ner-english'
    # bert_name = 'DeepPavlov/rubert-base-cased-conversational'
    # bert_name = 'dbmdz/bert-large-cased-finetuned-conll03-english'
    # bert_name = 'Jean-Baptiste/roberta-large-ner-english'
    # bert_name = 'swtb/XLM-RoBERTa-Base-Conll2003-English-NER-Finetune'
    # bert_name = 'yqelz/xml-roberta-large-ner-russian'

    # bert_name = 'bert-base-cased'

    bert_name = 'xlm-roberta-large'

    # bert_name = r"Z:\python-datasets\LitBank NER 2023\bert-base-cased_7_4"

    bert_name = r"Z:\python-datasets\LitBank NER 2023\bert-large-cased_7_4"

    bert_tuned = bert_name

    # bert_tuned = 'swtb/XLM-RoBERTa-Base-Conll2003-English-NER-Finetune'

    dts = DataTransform(model_name=bert_name, model_path=bert_tuned,
                        tokenizer=AutoTokenizer,
                        token_classification=AutoModelForTokenClassification
                        )

    # dts = DataTransform(model_name=bert_name, model_path=bert_tuned,
    #                     tokenizer=XLMRobertaTokenizer,
    #                     token_classification=XLMRobertaForSequenceClassification
    #                     )

    print(dts.idx2tag)
    print(dts.tag2idx)

    if 'PER' in dts.tag2idx:
        convert_function = convert_to_bio
    else:
        convert_function = convert_bio_tags

    dts.idx2tag = {0: 'O',
                   1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'}
    print(dts.idx2tag)

    dts.tag2idx = dict(sorted(dts.tag2idx.items(), key=lambda x: x[-1]))
    print(dts.tag2idx)
    print(dts.tag2idx.keys())

    # exit()

    train, sentences, labels, row_labels = dts.read_dataset('train_data_sent.csv')

    # train, sentences, labels, row_labels = dts.read_dataset('test_data_no_labels_sent.csv',
    #                                                         train_data=False)

    print(train)

    # print(list(sentences[0]))
    # print(list(labels[0]))
    # print(row_labels[0])

    sample = len(sentences)

    # sample = 1

    sentences = sentences[:sample]
    labels = labels[:sample]

    # idx_show = 0
    # print(len(sentences[idx_show]), list(sentences[idx_show]))
    # print(len(labels[idx_show]), list(labels[idx_show]))

    print('tag_values[:-1]', tag_values[:-1])

    true_labels = []
    for row_labels in labels:
        true_labels.extend(list(row_labels))

    predict_labels = []

    # sentences[0] = ['cold-and-gin', '""""', "'s", 'that', '--', 'well']
    # line = ' '.join(sentences[0])
    # line = re.sub(r'"+', '"', line)
    # line = re.sub(r'-+', '-', line)
    # sentences[0] = line.split()

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
        # if sample < 10:
        #     print(len(results), results)

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
