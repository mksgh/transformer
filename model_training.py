import time
import math
import torch
import torch.nn as nn
import torch.optim as optim

from training_util import train, evaluate
from tokens import pad_token, sos_token, unk_token, eos_token
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from custom_transformer import Transformer
from nlp_util import Multi30kDataset, tokenize_de, tokenize_en, create_vocab, translate_sentence


# setting seed value for reproducibility
seed = 247
torch.manual_seed(seed)


# data path
train_de_path = "./data/train.de.gz"
train_en_path = "./data/train.en.gz"
val_de_path = "./data/val.de.gz"
val_en_path = "./data/val.en.gz"
test_de_path = "./data/test_2016_flickr.de.gz"
test_en_path = "./data/test_2016_flickr.en.gz"


# creating tokens


# load datasets
train_data = Multi30kDataset(
    train_de_path, train_en_path, src_transform=tokenize_de, trg_transform=tokenize_en
)
val_data = Multi30kDataset(
    val_de_path, val_en_path, src_transform=tokenize_de, trg_transform=tokenize_en
)
test_data = Multi30kDataset(
    test_de_path, test_en_path, src_transform=tokenize_de, trg_transform=tokenize_en
)


# tokenize all train data
train_de_tokenized = [tokenize_de(sentence.strip()) for sentence in train_data.src_data]
train_en_tokenized = [tokenize_en(sentence.strip()) for sentence in train_data.trg_data]


# Create vocabularies with special tokens
src_vocab = create_vocab(
    train_de_tokenized, [pad_token, sos_token, eos_token, unk_token]
)
trg_vocab = create_vocab(
    train_en_tokenized, [pad_token, sos_token, eos_token, unk_token]
)


# Model hyperparameters
src_vocab_size = len(src_vocab)
trg_vocab_size = len(trg_vocab)
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
deopout = 0.1


# Initialize the model
model = Transformer(
    src_vocab_size,
    trg_vocab_size,
    d_model,
    num_heads,
    num_layers,
    d_ff,
    max_seq_length,
    deopout,
    src_vocab,
    trg_vocab,
)

print(
    f"The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
)

# Define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
pad_index = src_vocab[pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)


# Collate function for DataLoader
def collate_fn(batch):
    src_batch, trg_batch = [], []
    for sample in batch:
        src_batch.append(
            torch.tensor(
                [
                    src_vocab.get(token, src_vocab[unk_token])
                    for token in [sos_token] + sample["src"] + [eos_token]
                ]
            )
        )
        trg_batch.append(
            torch.tensor(
                [
                    trg_vocab.get(token, trg_vocab[unk_token])
                    for token in [sos_token] + sample["trg"] + [eos_token]
                ]
            )
        )

    src_batch = pad_sequence(src_batch, padding_value=src_vocab[pad_token])
    trg_batch = pad_sequence(trg_batch, padding_value=trg_vocab[pad_token])

    return src_batch.transpose(0, 1), trg_batch.transpose(0, 1)


# Training loop
n_epochs = 1
clip = 1.0
batch_size = 32

train_dataloader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn)

best_valid_loss = float("inf")

for epoch in range(n_epochs):
    start_time = time.time()

    train_loss = train(model, train_dataloader, optimizer, criterion, clip)
    valid_loss = evaluate(model, val_dataloader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "transformer-translation-model.pt")

    print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
    print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}")


# Load the best model for evaluation
model.load_state_dict(torch.load('transformer-translation-model.pt'))

# Example translations
for example_idx in range(3): 
    src = test_data[example_idx]['src']
    trg = test_data[example_idx]['trg']

    print(f'Source: {" ".join(src)}')
    print(f'Target: {" ".join(trg)}')

    translation = translate_sentence(" ".join(src), src_vocab, trg_vocab, model, torch.device('cpu' if torch.cuda.is_available() else 'cpu'))
    print(f'Predicted: {" ".join(translation)}')
    print()