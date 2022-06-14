import torch
from torchtext.data.metrics import bleu_score


def read_lines_from_file(file_path):
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines


def translate_sentence(model, sentence, src_vocab, trg_vocab, src_tokenizer,device, max_length=50):
    # print(sentence)

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in src_tokenizer.tokenizer(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # print(tokens)

    # sys.exit()
    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, "<sos>")
    tokens.append("<eos>")

    # Go through each german token and convert to an index
    text_to_indices = [ src_vocab.stoi[token] if token in src_vocab.stoi else src_vocab.stoi["<unk>"] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [trg_vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == trg_vocab.stoi["<eos>"]:
            break

    translated_sentence = [trg_vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]


def train(model, iterator, optimizer, criterion, clip, device):
    model.train()

    epoch_loss = 0

    for i, (src, trg) in enumerate(iterator):
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / (i + 1)


def evaluate(model, iterator, criterion, device):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / (i + 1)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def bleu(src_test, trg_test, model, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, device):
    targets = []
    outputs = []

    for i in range(len(src_test)):
        src = src_test[i]
        trg = trg_test[i]
        trg = [token.text.lower() for token in trg_tokenizer.tokenizer(trg)]
        prediction = translate_sentence(model, src, src_vocab, trg_vocab, src_tokenizer, device)
        prediction = prediction[:-1]  # remove <eos> token
        targets.append([trg])
        outputs.append(prediction)
    return bleu_score(outputs, targets) * 100
