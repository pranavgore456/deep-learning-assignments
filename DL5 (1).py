"""
DL Lab Assignment-5
Implement the Continuous Bag of Words (CBOW) Model.
Stages can be:
a. Data preparation
b. Generate training data
c. Train model
d. Output
"""
import re
sentences = """    We are ?????about$$$$$\ to 01930183study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.  
"""
#Clean Data
# remove special characters
sentences = re.sub('[^A-Za-z]+', ' ', sentences)
sentences = re.sub(r'(?:^| )\w(?:$| )', ' ', sentences).strip()

# lower all characters
sentences = sentences.lower()
# Vocabulary
words = sentences.split()

data = []
for i in range(2, len(words) - 2):
    context = [words[i - 2], words[i - 1], words[i + 1], words[i + 2]]
    target = words[i]
    data.append((context, target))
print(data)