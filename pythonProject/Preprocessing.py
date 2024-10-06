import inflect
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

p = inflect.engine()

def replace_numbers(text):
    words = text.split()
    for i, word in enumerate(words):

        if ',' in word:
            try:
                parts = word.split(',')
                num = int(''.join(parts))
                words[i] = p.number_to_words(num)
            except ValueError:
                pass

        if '.' in word:
            try:
                parts = word.split('.')
                whole_part = int(parts[0])
                decimal_part = parts[1]
                words[i] = p.number_to_words(whole_part) + ' point ' + p.number_to_words(int(decimal_part))
            except ValueError:
                pass

        else:
            try:
                num = int(word)
                words[i] = p.number_to_words(num)
            except ValueError:
                pass

    return ' '.join(words)


if __name__ == '__main__':

    requirements_texts = [
        "We want a website where customers can browse products by category, filter by price, and add items to their cart. Users should be able to create accounts and check out with PayPal or credit card.",
        "I need an app that helps users track their daily steps and calories. It should send notifications to remind users to drink water, and it should show progress graphs weekly and monthly.",
        "We want to build a system where multiple users can write and edit articles, and an admin can approve them before publishing. It should support different formats like text, images, and videos.",
        "The system must support user login and authentication.",
        "Response time should be under 2 seconds.",
        "Data encryption is mandatory.",
        "The application must allow for real-time data updates.",
        "The system should be scalable to handle 10,000 concurrent users."
    ]

    for i, text in enumerate(requirements_texts):
        requirements_texts[i] = replace_numbers(text)

    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(requirements_texts)

    tfidf_matrix = vectorizer.transform(requirements_texts)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    print("IMPORTANCE MATRIX")
    print(tfidf_df)
    print("VOCABULARY")
    print(vectorizer.get_feature_names_out())

