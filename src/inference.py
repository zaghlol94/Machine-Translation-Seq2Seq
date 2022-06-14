import torch
import pickle
import spacy
import argparse
from utils import translate_sentence
from config import config
from model import Encoder, Decoder, Seq2Seq

parser = argparse.ArgumentParser(description="translate string from german to english")
parser.add_argument("-s", "--sentence", type=str, required=True, help="sentence that you want to translate")
args = parser.parse_args()

print(args.sentence)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

src_tokenizer = spacy.load('de_core_news_sm')
trg_tokenizer = spacy.load('en_core_web_sm')

with open('src_vocab.pkl', 'rb') as inp:
    src_vocab = pickle.load(inp)
with open('trg_vocab.pkl', 'rb') as inp:
    trg_vocab = pickle.load(inp)

INPUT_DIM = len(src_vocab.stoi)
OUTPUT_DIM = len(trg_vocab.stoi)

enc = Encoder(INPUT_DIM, config["ENC_EMB_DIM"], config["HID_DIM"], config["N_LAYERS"], config["ENC_DROPOUT"])
dec = Decoder(OUTPUT_DIM, config["DEC_EMB_DIM"], config["HID_DIM"], config["N_LAYERS"], config["DEC_DROPOUT"])

model = Seq2Seq(enc, dec, device).to(device)

model.load_state_dict(torch.load(config["test_config"]["model_path"]))
print(" ".join(translate_sentence(
        model, args.sentence, src_vocab, trg_vocab, src_tokenizer, device, max_length=50
    )[:-1]))
