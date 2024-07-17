# LitBank-NER-2023
Решение задачи извлечения именованных сущностей NER.   
https://www.kaggle.com/competitions/litbank-ner-2024/overview  

train-bert-models.ipynb - тетрадка для обучения модели  
data_process.py - класс и вспомогательные функции для обработки данных  
df_addons.py - вспомогательный модель для красивостей  
predict.py - предсказание на теством файле test_data_no_labels_sent.csv  
merge_submit.py - объединение предсказаний нескольких моделей (предсказания нужно поместить в каталог .\compare\*.csv  
