import string
from typing import List, Tuple
from collections import Counter
import random
import pickle

from nltk import word_tokenize


def remove_punct(sentence):
    punct = set(string.punctuation)
    return ''.join([ch for ch in sentence if ch not in punct])


def tokenize_sentence(sentence, tokenizer=word_tokenize) -> List[str]:
    return tokenizer(sentence, language='english')


def sentence_to_integers(sentence: List[str], vocab_list: List[str]) -> List[int]:
    data = []
    for t in sentence:
        if t in vocab_list:
            word_num = vocab_list.index(t) + 1
        else:
            word_num = 0
        data.append(word_num)
    return data


class BagOfWords:

    def __init__(self, vocab_size: int = 10_000, tokenizer=None):
        self._vocab_size = vocab_size
        self._raw_vocab: Counter = Counter()
        self._labels = set()
        self._data_set: List[Tuple[int, List[int]]] = []
        self._raw_sentences: List[Tuple[str, List[str]]] = []
        self._loaded_vocab: List[str] = []
        self.__word_tokenizer = tokenizer if tokenizer else word_tokenize

    def load_newline_txt(self, label: str, filepath: str, encoding: str = 'utf-8') -> None:
        self._labels.add(label)
        with open(filepath, 'r', encoding=encoding) as entry_file:
            for line in entry_file:
                tmp = self.__word_tokenizer(remove_punct(line.lower()))
                self._raw_sentences.append((label, tmp))
        return

    def _word_count(self):
        for sent in self._raw_sentences:
            _, words = sent
            for w in words:
                if w in self._raw_vocab:
                    self._raw_vocab[w] += 1
                else:
                    self._raw_vocab[w] = 1

    def prepare_data(self):
        self._word_count()
        self._loaded_vocab = [i for i, _ in self._raw_vocab.most_common(self._vocab_size-1)]
        integer_labels = {k: v for v, k in enumerate(sorted(self._labels))}
        for sent in self._raw_sentences:
            label, text = sent
            label = integer_labels[label]
            word_indices = sentence_to_integers(text, self._loaded_vocab)
            self._data_set.append((label, word_indices))

    def test_train_split(self, test_split: float = 0.1):
        random.shuffle(self._data_set)
        split = int(len(self._data_set) * test_split)
        test = self._data_set[:split]
        train = self._data_set[split:]
        test_labels = [lab for lab, _ in test]
        test_values = [val for _, val in test]
        train_labels = [lab for lab, _ in train]
        train_values = [val for _, val in train]
        return (train_values, train_labels), (test_values, test_labels)

    def sentence_to_data(self, sentence: str):
        tokens = self.__word_tokenizer(sentence)
        return sentence_to_integers(tokens, self._loaded_vocab)

    def save_vocab(self, file_path: str):
        vocab_only = self._loaded_vocab
        with open(file_path, 'wb') as outfile:
            pickle.dump(vocab_only, outfile)

    def load_vocab_labels(self, file_path: str, labels: List[str]):
        with open(file_path, 'rb') as infile:
            self._loaded_vocab = pickle.load(infile)
        for lab in labels:
            self._labels.add(lab)


if __name__ == '__main__':
    bag = BagOfWords()
    bag.load_newline_txt('russian', 'russian.txt')
    bag.load_newline_txt('bulgarian', 'bulgarian.txt')
    bag.prepare_data()
    train, test = bag.test_train_split()
    print(train)