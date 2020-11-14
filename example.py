import tensorflow as tf

import csv
import glob
import unicodedata
import numpy as np
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from pymystem3 import Mystem
from string import punctuation
russian_stopwords = set(stopwords.words('russian'))

num_epochs = 10
vocab_size = 150
embedding_dim = 32
max_length = 30
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

articles = []
labels = []

""" train_file_path = tf.keras.utils.get_file("bbc-text.csv", "https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv") """
import_training_path = "C:/Pogromirovanie/TensorBot/training_data/все_вопросы(4).csv"
""" all_training_csv = glob.glob(import_training_path + "/*.csv") """

with open(import_training_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        row[1].encode('utf-8')
        labels.append(row[0])
        article = row[1]
        for word in russian_stopwords:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        articles.append(article)
print("Общее количество вариантов ответов - ", len(labels))
print("Общее количество вопросов - ", len(articles))

print("Первый элемент вопросов", articles[0])

train_size = int(len(articles) * training_portion)

train_articles = articles[0: train_size]
train_labels = labels[0: train_size]


validation_articles = articles[train_size:]
validation_labels = labels[train_size:]
articles = articles[:train_size]
labels  = labels[:train_size]

print("Тренировочный размер - ", train_size)
print("Количество тренировочных вопросов - ", len(train_articles))
print("Количество тренировочных вариантов ответов - ", len(train_labels))
print("Количество тестовых вопросов - ", len(validation_articles))
print("Количество тестовых вариантов ответов - ", len(validation_labels))


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index

dict(list(word_index.items())[0:10])

train_sequences = tokenizer.texts_to_sequences(train_articles)

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print("Количество тестовых последовательностей", len(validation_sequences))
print("Validation shape - ", validation_padded.shape)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_article(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
print(decode_article(train_padded[10]))
print('---')
print(train_articles[10])

model = tf.keras.Sequential([
    # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    # use ReLU in place of tanh function since they are very good alternatives of each other.
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # Add a Dense layer with 6 units and softmax activation.
    # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
    tf.keras.layers.Dense(5, activation='softmax')
])
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(training_label_seq[0], validation_label_seq[0])
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

labels = ['Периоды', 'Номер справки', 'Поиск товаров', 'Вакансия']


txt = ["периоды"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)[0]
print("Вопрос - ", txt)
print(pred)
print("Предсказанный моделью ID - ", labels[np.argmax(pred)])

txt = ["Что данные телефона справочной информации"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)
print("Вопрос - ", txt)
print(pred)
print("Предсказанный моделью ID - ", labels[np.argmax(pred)])

txt = ["хочу найти товар"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)
print("Вопрос - ", txt)
print(pred)
print("Предсказанный моделью ID - ", labels[np.argmax(pred)])

txt = ["Проясни што такое периоды"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)
print("Вопрос - ", txt)
print(pred)
print("Предсказанный моделью ID - ", labels[np.argmax(pred)])

txt = ["Хочу через тебя набросать резюме"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)
print("Вопрос - ", txt)
print(pred)
print("Предсказанный моделью ID - ", labels[np.argmax(pred)])