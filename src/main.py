import asyncio
import pandas as pd
from flask_limiter import Limiter
from flask import Flask, request, jsonify
from flask_limiter.util import get_remote_address
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["10001 per day", "31 per hour"],
    storage_uri="memory://",
)


def read_data():
    try:
        # Load the data from the CSV file
        df = pd.read_csv('data/Financebench.csv')

        # Prepare the data
        Question = df['question']
        Answer = df['answer']
        return Question, Answer
    except KeyError as ke:
        raise ke


def fit_model(Question, Answer):
    try:
        global model, vectorizer
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            Question, Answer, test_size=0.2)

        # Create a CountVectorizer object
        vectorizer = CountVectorizer()

        # Fit the vectorizer to the training data
        X_train_vectorized = vectorizer.fit_transform(X_train)

        # Create a LogisticRegression object
        model = LogisticRegression()

        # Train the model on the training data
        model.fit(X_train_vectorized, y_train)

        # Evaluate the model on the testing data
        X_test_vectorized = vectorizer.transform(X_test)

        # if you want to predict the accuracy, print score
        score = model.score(X_test_vectorized, y_test)
    except Exception as e:
        raise e


async def answer(question):
    new_question_vectorized = vectorizer.transform([question])
    answer = model.predict(new_question_vectorized)[0]
    return answer


@app.route("/chat/", methods=["POST"])
@limiter.limit("10/second", override_defaults=False)
async def get_answer():
    content = request.json
    questions = content['question']
    # you can send questions as a list of questions or just one question in string format.
    if type(questions) == list:
        predicted_answer = await asyncio.gather(*[answer(question) for question in questions])
    else:
        predicted_answer = await answer(questions)
    return jsonify({"answer": predicted_answer})

if __name__ == "__main__":
    try:
        print("[*] reading data...\\")
        Question, Answer = read_data()
        print("[*] loading data...\\")
        fit_model(Question=Question, Answer=Answer)
        print("[*] data onboard complete")
    except KeyError as e:
        raise e
    print("[*] initiate server")
    app.run(host="0.0.0.0", port=8001, debug=True, threaded=True)
