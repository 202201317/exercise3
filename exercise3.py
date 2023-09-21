import spacy
import matplotlib.pyplot as plt
from collections import Counter

# Loading Spacy's English Model
nlp = spacy.load("en_core_web_sm")

# Reading Moby Dick files
with open('mobydick.txt', 'r', encoding='utf-8') as file:
    moby_dick_text = file.read()

# Process Text
doc = nlp(moby_dick_text)

# Tokenization
tokens = [token.text for token in doc]

# Stop-words filtering
filtered_tokens = [token for token in tokens if not token.lower() in nlp.Defaults.stop_words]

# Parts-of-Speech (POS) tagging and frequency
pos_counts = Counter([token.pos_ for token in doc])
top_pos = pos_counts.most_common(5)

print("Top 5 Parts of Speech and Their Frequencies:")
for pos, count in top_pos:
    print(f"{pos}: {count}")

# Lemmatization
lemmatized_tokens = [token.lemma_ for token in doc[:20]]

print("\nTop 20 Lemmatized Tokens:")
print(lemmatized_tokens)

# Plotting frequency distribution
pos_labels, pos_freq = zip(*pos_counts.items())
plt.bar(pos_labels, pos_freq)
plt.title("POS Frequency Distribution")
plt.xlabel("POS")
plt.ylabel("Frequency")
plt.show()