import pickle

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# simple skill list
skills = ["python", "java", "machine learning", "data science", "sql"]

def check_skills(text):
    found = []
    for skill in skills:
        if skill in text.lower():
            found.append(skill)
    return found

def predict_resume(text):
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()

    found_skills = check_skills(text)

    return prediction, round(prob * 100, 2), found_skills
