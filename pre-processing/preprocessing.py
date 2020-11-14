""" Препроцессинг обучающей/тестовой выборки """
import csv
import datetime

from pymystem3 import Mystem
from string import punctuation
from nltk.corpus import stopwords

import_training_path = "C:/Pogromirovanie/TensorBot/training_data/все_вопросы(4).csv"
start_time = datetime.datetime.now()

print("--------------------------------------")
print("Начинаю препроцессинг")
print("Время начала - ", start_time)
print("--------------------------------------")

russian_stopwords = set(stopwords.words('russian'))

labels = []
questions = []

print("--------------------------------------")
print("Считываю данные файла с путем ", import_training_path)
print("--------------------------------------")

with open(import_training_path, 'r', encoding='utf8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        row[1].encode('utf-8')
        labels.append(row[0])
        question = row[1]
        for word in russian_stopwords:
            token = ' ' + word + ' '
            question = question.replace(token, ' ')
            question = question.replace(' ', ' ')
        questions.append(question)
print("Общее количество вариантов ответов - ", len(labels))
print("Общее количество вопросов - ", len(questions))

print("--------------------------------------")
print("Провожу лемматизацию")
print("--------------------------------------")

mystem = Mystem()
word_dictionary = {}
def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
                and token != " " \
                and token.strip() not in punctuation]

    new_tokens = []
    for token in tokens:
        if token in word_dictionary:
            word_dictionary[token] += 1
        else:
            word_dictionary[token] = 1
        
        """ if (token[-2] == 'т' or token[-2] == 'ч') and token[-1] == 'ь':
            continue
        if token[-2] == 'с' and token[-1] == 'я':
            continue
        new_tokens.append(token) """

    text = " ".join(tokens)

    return text

k = 1
preprocessed_questions = []
for i in questions:
    processed_question = preprocess_text(i)

    """ if processed_question in preprocessed_questions:
        continue """

    print(k, processed_question)
    preprocessed_questions.append(processed_question)
    k += 1

print("В словаре находятся следующие слова в следующем количестве -", word_dictionary)

print("--------------------------------------")
print("Записываю результаты")
print("--------------------------------------")

with open('all_questions_without_delete_preprocessed(4).csv', 'w', newline='') as file:
    writer = csv.writer(file)
    i = 0
    writer.writerow(['label', 'question'])
    while i < len(preprocessed_questions):
        writer.writerow([labels[i], preprocessed_questions[i]])
        i += 1

end_time = datetime.datetime.now()
passed_time = end_time - start_time
print("--------------------------------------")
print("Препроцессинг завершен")
print("Время окончания - ", end_time)
print("Этап занял ", passed_time.seconds//3600, " часов, ", (passed_time.seconds//60)%60, " минут и ", passed_time.seconds, " секунд")
print("--------------------------------------")