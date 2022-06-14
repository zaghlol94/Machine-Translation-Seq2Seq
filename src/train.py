import spacy
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import time
import math
from utils import read_lines_from_file, train, evaluate, epoch_time, count_parameters
from model import Encoder, Decoder, Seq2Seq
from vocabulary import Vocabulary
from dataset import get_loader
from config import config

src_train = read_lines_from_file(config["src_train"])
trg_train = read_lines_from_file(config["trg_train"])
src_valid = read_lines_from_file(config["src_valid"])
trg_valid = read_lines_from_file(config["trg_valid"])


print(f"Number of training examples: {len(src_train)}")
print(f"Number of validation examples: {len(src_valid)}")

src_tokenizer = spacy.load('de_core_news_sm')
trg_tokenizer = spacy.load('en_core_web_sm')

src_vocab = Vocabulary(2, src_tokenizer)
src_vocab.build_vocabulary(src_train)

trg_vocab = Vocabulary(2, trg_tokenizer)
trg_vocab.build_vocabulary(trg_train)

with open('src_vocab.pkl', 'wb') as file:
    pickle.dump(src_vocab, file, pickle.HIGHEST_PROTOCOL)

with open('trg_vocab.pkl', 'wb') as file:
    pickle.dump(trg_vocab, file, pickle.HIGHEST_PROTOCOL)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Unique tokens in source (de) vocabulary: {len(src_vocab.stoi)}")
print(f"Unique tokens in target (en) vocabulary: {len(trg_vocab.stoi)}")

train_loader, train_dataset = get_loader(config["src_train"], config["trg_train"], src_vocab, trg_vocab)
val_loader, val_dataset = get_loader(config["src_valid"], config["trg_valid"], src_vocab, trg_vocab)
test_loader, test_dataset = get_loader(config["test_config"]["src_test"], config["test_config"]["trg_test"], src_vocab,
                                       trg_vocab)

INPUT_DIM = len(src_vocab.stoi)
OUTPUT_DIM = len(trg_vocab.stoi)

CLIP = 1
learning_rate = 0.001

enc = Encoder(INPUT_DIM, config["ENC_EMB_DIM"], config["HID_DIM"], config["N_LAYERS"], config["ENC_DROPOUT"])
dec = Decoder(OUTPUT_DIM, config["DEC_EMB_DIM"], config["HID_DIM"], config["N_LAYERS"], config["DEC_DROPOUT"])

model = Seq2Seq(enc, dec, device).to(device)
print(model)
print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
TRG_PAD_IDX = trg_vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

best_valid_loss = float('inf')

for epoch in range(config["N_EPOCHS"]):

    start_time = time.time()
    train_loader, train_dataset = get_loader(config["src_train"], config["trg_train"], src_vocab, trg_vocab)
    val_loader, val_dataset = get_loader(config["src_valid"], config["trg_valid"], src_vocab, trg_vocab)

    train_loss = train(model, train_loader, optimizer, criterion, CLIP, device)
    valid_loss = evaluate(model, val_loader, criterion, device)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
