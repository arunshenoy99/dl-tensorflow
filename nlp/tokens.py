import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

tokenizer = Tokenizer(num_words = 100, oov_token = '<OOV>') # num_words gives number of words in a sentence to be safe 100, oov token is used when words out of vocabulary are used
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
test_sequences = tokenizer.texts_to_sequences(test_data)
print(word_index)
print(sequences)
print(test_sequences)