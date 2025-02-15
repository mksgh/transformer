
import torch
import spacy
import gzip
from tokens import sos_token, unk_token, eos_token
from torch.utils.data import Dataset

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

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_sentence = self.src_data[idx].strip()
        trg_sentence = self.trg_data[idx].strip()

        if self.src_transform:
            src_sentence = self.src_transform(src_sentence)
        if self.trg_transform:
            trg_sentence = self.trg_transform(trg_sentence)

        return {"src": src_sentence, "trg": trg_sentence}

def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()

    tokens = [sos_token] + tokenize_de(sentence) + [eos_token]

    src_indexes = [src_vocab.get(token, src_vocab[unk_token]) for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.generate_mask(src_tensor, src_tensor)

    with torch.no_grad():
        enc_src = model.encoder_embedding(src_tensor)
        for enc_layer in model.encoder_layers:
            enc_src = enc_layer(enc_src, src_mask[0])

    trg_indexes = [trg_vocab[sos_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.generate_mask(src_tensor, trg_tensor)

        with torch.no_grad():
            output = model.decoder_embedding(trg_tensor)
            for dec_layer in model.decoder_layers:
                output = dec_layer(output, enc_src, src_mask[0], trg_mask[1])
            output = model.fc_out(output)

        pred_token = output.argmax(2)[:,-1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_vocab[eos_token]:
            break

    trg_tokens = [list(trg_vocab.keys())[list(trg_vocab.values()).index(i)] for i in trg_indexes]

    return trg_tokens[1:-1]