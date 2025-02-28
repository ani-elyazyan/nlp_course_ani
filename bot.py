import json
import os
import requests
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import numpy as np
import gensim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

!pip install nltk scikit-learn aiogram

import json

!wget faq.json https://raw.githubusercontent.com/vifirsanova/compling/main/tasks/task3/faq.json
with open('faq.json', encoding='utf-8') as f:
  data = json.load(f)
data

dp = Dispatcher()
bot = Bot(token='token')

# Загрузка файла 
!wget faq.json https://raw.githubusercontent.com/vifirsanova/compling/main/tasks/task3/faq.json

with open("faq.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Извлекаем вопросы и ответы из данных
faq_questions = []
faq_answers = []
for q in data.values():
    for y in q:
        faq_questions.append(y['question'])
        faq_answers.append(y['answer'])
        
        
# метод TF-IDF 
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(faq_questions)

# метод Word2Vec
sentences = [q.split() for q in faq_questions]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Функция для усреднения векторов слов в вопросе
def sentence_vector(sentence, model):
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0)

# Векторизуем вопросы
faq_vectors = np.array([sentence_vector(q, model) for q in faq_questions])

# Функция для выбора наиболее подходящего ответа на основе TF-IDF
def get_answer_tfidf(question):
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, tfidf_matrix)
    best_match_idx = similarities.argmax()
    return faq_answers[best_match_idx]

# Функция для выбора наиболее подходящего ответа на основе Word2Vec
def get_answer_word2vec(question):
    question_vector = sentence_vector(question, model).reshape(1, -1)
    similarities = cosine_similarity(question_vector, faq_vectors)
    best_match_idx = similarities.argmax()
    return faq_answers[best_match_idx]

# Обработка команды /start
@dp.message(Command("start"))
async def start_command(message: types.Message):
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="О компании")],
            [KeyboardButton(text="Пожаловаться")]
        ],
        resize_keyboard=True
    )
    await message.answer("Привет! Я бот, который может ответить на ваши вопросы.", reply_markup=keyboard)

# Обработка кнопки "О компании"
@dp.message(lambda message: message.text == "О компании")
async def about_company(message: types.Message):
    await message.answer("Наша компания занимается доставкой товаров по всей стране.")

# Обработка кнопки "Пожаловаться"
@dp.message(lambda message: message.text == "Пожаловаться")
async def complain(message: types.Message):
    await message.answer("Пожалуйста, отправьте изображение.")

# Обработка изображений
@dp.message(lambda message: message.content_type == "photo")
async def handle_photo(message: types.Message):
    file_id = message.photo[-1].file_id
    file = await bot.get_file(file_id)
    filename = file.file_path.split("/")[-1]
    filesize = message.photo[0].file_size
    await message.answer(f'Ваш запрос передан специалисту. Название файла: {filename}, размер: {filesize} байт')

# Обработка вопросов пользователей
@dp.message()
async def answer_question(message: types.Message):
    question = message.text
    answer_tfidf = get_answer_tfidf(question)  
    answer_word2vec = get_answer_word2vec(question)  
    await message.answer("Ответ на основе TF-IDF: " + answer_tfidf)
    await message.answer("Ответ на основе Word2Vec: " + answer_word2vec)


# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    await main()
