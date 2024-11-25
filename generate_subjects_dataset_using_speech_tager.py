import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Define the sentence
sentence = "Calling someone out in a movie theater"

# Process the sentence
doc = nlp(sentence)

# Find the subject
for token in doc:
    print(token.text, token.pos_)


