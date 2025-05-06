from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from langdetect import detect
from googletrans import Translator

# Expanded multilingual travel FAQ
faq = {
    "what tours are available": "We offer city tours, island excursions, and cultural experiences.",
    "do you offer airport transfers": "Yes, private and group airport transfers are available upon request.",
    "how do i book a tour": "You can book online through our platform or contact our support team.",
    "what languages are tours offered in": "Tours are typically available in English, Greek, Russian, Romanian, and more.",
    "can i cancel or reschedule": "Yes, most bookings can be modified or cancelled according to our policy.",
    "what are family-friendly options": "Family tours include beach days, zoo visits, aquariums, and easy sightseeing.",
    "do you organize cruises": "Yes, we offer full-day and sunset cruises to islands and along the coast.",
    "are meals included": "Some tours include meals or tastings. Please check the tour description for details.",
    "do i need entry tickets for sites": "Entry tickets may be included or added separately depending on the tour.",
    "how can i contact support": "Reach out via our contact form, WhatsApp, or call the support number."
}

questions = list(faq.keys())
answers = list(faq.values())

vectorizer = TfidfVectorizer().fit(questions)
faq_vectors = vectorizer.transform(questions)
translator = Translator()

def smart_travel_bot(user_input):
    try:
        lang = detect(user_input)
    except:
        lang = 'en'

    if lang != 'en':
        translated_input = translator.translate(user_input, src=lang, dest='en').text
    else:
        translated_input = user_input

    user_vec = vectorizer.transform([translated_input])
    similarity = cosine_similarity(user_vec, faq_vectors)
    best_match_idx = similarity.argmax()
    best_question = questions[best_match_idx]
    score = similarity[0][best_match_idx]
    fuzzy_score = fuzz.ratio(translated_input.lower(), best_question.lower())

    if score > 0.4 or fuzzy_score > 70:
        answer = faq[best_question]
    else:
        answer = "I'm sorry, I couldn't find an answer to that. Please contact our support team."

    if lang != 'en':
        answer = translator.translate(answer, src='en', dest=lang).text

    return answer

def run_chat():
    print("🤖 SmartTravelBot is online! Ask anything about travel. Type 'exit' to stop.")
    while True:
        user = input("🧳 You: ")
        if user.lower() in ['exit', 'quit']:
            print("🤖 Safe travels! 👋")
            break
        print("🤖", smart_travel_bot(user))
