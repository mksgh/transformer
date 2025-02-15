import os
import spacy
import gzip
from torch.utils.data import Dataset, DataLoader

# # downloading required spacy model
# os.system("python -m spacy download en_core_web_sm")
# os.system("python -m spacy download de_core_news_sm")

# loading the spacy model
spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")


def tokenize_de(text):
    return [token.text.lower() for token in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [token.text.lower() for token in spacy_en.tokenizer(text)]


def create_vocab(tokenized_sentences, special_tokens):
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    for sentence in tokenized_sentences:
        for token in sentence:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab

class Multi30kDataset(Dataset):
    def __init__(self, src_file, trg_file, src_transform=None, trg_transform=None):
        self.src_data = self.load_data(src_file)
        self.trg_data = self.load_data(trg_file)
        self.src_transform = src_transform
        self.trg_transform = trg_transform

    def load_data(self, file_path):
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            data = f.readlines()
        return data

    # def __len__(self):
    #     return len(self.src_data)

    # def __getitem__(self, idx):
    #     src_sentence = self.src_data[idx].strip()
    #     trg_sentence = self.trg_data[idx].strip()

    #     if self.src_transform:
    #         src_sentence = self.src_transform(src_sentence)
    #     if self.trg_transform:
    #         trg_sentence = self.trg_transform(trg_sentence)

    #     return {"src": src_sentence, "trg": trg_sentence}
