import torch
import torch.nn as nn
import pickle
import math
import spacy
from utils import read_lines_from_file, evaluate, bleu
from dataset import get_loader
from config import config
from model import Encoder, Decoder, Seq2Seq

src_test = read_lines_from_file(config["test_config"]["src_test"])
trg_test = read_lines_from_file(config["test_config"]["trg_test"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

src_tokenizer = spacy.load('de_core_news_sm')
trg_tokenizer = spacy.load('en_core_web_sm')

with open('src_vocab.pkl', 'rb') as inp:
    src_vocab = pickle.load(inp)
with open('trg_vocab.pkl', 'rb') as inp:
    trg_vocab = pickle.load(inp)

test_loader, test_dataset = get_loader(config["test_config"]["src_test"], config["test_config"]["trg_test"], src_vocab,
                                       trg_vocab)

TRG_PAD_IDX = trg_vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

INPUT_DIM = len(src_vocab.stoi)
OUTPUT_DIM = len(trg_vocab.stoi)

enc = Encoder(INPUT_DIM, config["ENC_EMB_DIM"], config["HID_DIM"], config["N_LAYERS"], config["ENC_DROPOUT"])
dec = Decoder(OUTPUT_DIM, config["DEC_EMB_DIM"], config["HID_DIM"], config["N_LAYERS"], config["DEC_DROPOUT"])

model = Seq2Seq(enc, dec, device).to(device)

model.load_state_dict(torch.load(config["test_config"]["model_path"]))

test_loss = evaluate(model, test_loader, criterion, device)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
print("Belu Score: ",
      bleu(src_test[:100], trg_test[:100], model, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, device))
