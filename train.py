import tensorflow as tf
import numpy as np
import sentencepiece as spm
import fasttext
import tokenizer
import numpy as np
from tqdm import tqdm
import math

EMBEDDING_DIM = 100 #default value of fattext word vector

def parse_corpus_line(line):
    _, sentence, label = line.strip().split('\t')
    label = int(label)
    return sentence, label

def read_corpus(path):
    sentences = []
    labels = []

    for line in open(path):
        sentence, label = parse_corpus_line(line)
        sentences.append(sentence)
        labels.append(label)

    return np.array(sentences), np.array(labels)

def pad(data, max_len=0):    
    if max_len == 0:
        max_len = max(len(tokens) for tokens in data)

    result = []
    for tokens in tqdm(data, desc='Padding'):
        if len(tokens) >= max_len: #Truncate if tokens are longer than max_len
            result.append(tokens[:max_len])
        else:
            n_to_pad = max_len - len(tokens) 
            result.append(tokens + [''] * n_to_pad)

    return max_len, result

def preprocess(sentences, tokenize_method, unit):
    '''
    This function does the following for each sentence
    1. Tokenize per method and unit
    2. Padding
    '''
    tokenize_func = tokenizer.get_tokenizer(tokenize_method, unit)    
    tokenized_sentences = [tokenize_func(sentence) for sentence in tqdm(sentences, desc='Tokenizing')]    
    max_tokens, padded_sentences = pad(tokenized_sentences)
    
    return padded_sentences

class Dataset(tf.keras.utils.Sequence):
    fasttext_model_cache = {}
    
    def __init__(self, x_set, y_set, batch_size, tokenize_method, unit):
        self.x_set = x_set
        self.y_set = y_set
        self.batch_size = batch_size

        fasttext_model_path = 'model/ft.{}.{}.bin'.format(tokenize_method, unit)        
        if fasttext_model_path not in Dataset.fasttext_model_cache:
            Dataset.fasttext_model_cache[fasttext_model_path] = fasttext.load_model(fasttext_model_path)        
        self.fasttext_model = Dataset.fasttext_model_cache[fasttext_model_path]

    def __len__(self):
        return math.ceil(len(self.x_set) / self.batch_size)

    def __getitem__(self, idx):
        padded_sentences = self.x_set[idx * self.batch_size:(idx + 1) * self.batch_size]        
        word_vectors = [self.get_word_vectors(padded_sentence) for padded_sentence in padded_sentences]        
        
        batch_y = self.y_set[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(word_vectors), np.array(batch_y)
    
    def get_word_vectors(self, words):
        result = []
        for word in words:
            if not word: # zeros for padding
                result.append(np.zeros((EMBEDDING_DIM,)))
            else:
                result.append(self.fasttext_model.get_word_vector(word))

        return np.array(result)

def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), 
                                            input_shape=(None, EMBEDDING_DIM)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))    

    return model

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()               
    parser.add_argument('train')
    parser.add_argument('test')
    parser.add_argument('--method', choices=['eojeol', 'subword', 'morpheme'], required=True)
    parser.add_argument('--unit', choices=['jaso', 'char'], required=True)
    parser.add_argument('--batch-size', type=int, default=128)            
    parser.add_argument('--test-batch-size', type=int)
    parser.add_argument('--epochs', type=int, default=10)    
    args = parser.parse_args()
    
    train_sentences, train_labels = read_corpus(args.train)    
    train_padded_sentences = preprocess(train_sentences, args.method, args.unit)
    train_dataset = Dataset(train_padded_sentences, train_labels, args.batch_size, args.method, args.unit)     

    model = build_model()    
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    model.fit(train_dataset, epochs=args.epochs)
    model.save('model/classfier.{}.{}.model'.format(args.method, args.unit))

    test_sentences, test_labels = read_corpus(args.test)
    test_padded_sentences = preprocess(test_sentences, args.method, args.unit)       
    test_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size    
    test_dataset = Dataset(test_padded_sentences, test_labels, test_batch_size, args.method, args.unit)    
    
    test_loss, test_accuracy = model.evaluate(test_dataset)    
    print('test_loss', test_loss)
    print('test_accuracy', test_accuracy)
