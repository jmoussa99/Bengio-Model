import argparse
import csv
import json
import math
import os
import random
import time

import torch
import torch.nn as nn

def read_corpus(file_name,vocab,words,corpus,threshold):
    
    wID = len(vocab)
    
    if threshold > -1:
        with open(file_name,'rt') as f:
            for line in f:
                line = line.replace('\n','')
                tokens = line.split(' ')
                for t in tokens:
                    try:
                        elem = words[t]
                    except:
                        elem = [wID,0]
                        vocab.append(t)
                        wID = wID + 1
                    elem[1] = elem[1] + 1
                    words[t] = elem

        temp = words
        words = {}
        vocab = []
        wID = 0
        words['<unk>'] = [wID,100]
        vocab.append('<unk>')
        for t in temp:
            if temp[t][1] >= threshold:
                vocab.append(t)
                wID = wID + 1
                words[t] = [wID,temp[t][1]]
           
                    
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(' ')
            for t in tokens:
                try:
                    wID = words[t][0]
                except:
                    wID = words['<unk>'][0]
                corpus.append(wID)
                
    return [vocab,words,corpus]

def encode(text,words):
        encoded = []
        tokens = text.split(' ')
        for i in range(len(tokens)):
            try:
                wID = words[tokens[i]][0]
            except:
                wID = words['<unk>'][0]
            encoded.append(wID)
        return encoded
            
class bengio(torch.nn.Module):
    def __init__(
        self,
        dim=100,
        window=5,
        batchsize=1024,
        vocab_size=33279,
        hidden_dim=100,
        activation=torch.tanh,
    ):
        super().__init__()
        self.dim = dim
        self.window = window
        self.batchsize = batchsize
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.embedding = nn.Embedding(vocab_size, dim)
        self.hidden = nn.Linear(window * dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embeddings = self.embedding(x)
        embeddings = embeddings.reshape(embeddings.shape[0], self.window * self.dim)
        hidden = self.activation(self.hidden(embeddings))
        return self.output(hidden)


def manual_cross_entropy(logits, targets):
    """Average negative log likelihood, implemented without nn.CrossEntropyLoss."""
    max_logits = logits.max(dim=1, keepdim=True).values
    shifted_logits = logits - max_logits
    log_denominator = max_logits.squeeze(1) + torch.log(torch.exp(shifted_logits).sum(dim=1))
    target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    return (log_denominator - target_logits).mean()


def batch_iter(corpus, window, batchsize, shuffle=False):
    num_examples = corpus.numel() - window
    if num_examples <= 0:
        return

    device = corpus.device
    offsets = torch.arange(window, device=device).unsqueeze(0)

    if shuffle:
        order = torch.randperm(num_examples, device=device)
        for start in range(0, num_examples, batchsize):
            starts = order[start:start + batchsize]
            contexts = corpus[starts.unsqueeze(1) + offsets]
            targets = corpus[starts + window]
            yield contexts, targets
    else:
        for start in range(0, num_examples, batchsize):
            end = min(start + batchsize, num_examples)
            starts = torch.arange(start, end, device=device)
            contexts = corpus[starts.unsqueeze(1) + offsets]
            targets = corpus[starts + window]
            yield contexts, targets


def evaluate(model, corpus, opt, split_name='valid'):
    model.eval()
    total_loss = 0.0
    total_words = 0

    with torch.no_grad():
        for contexts, targets in batch_iter(corpus, opt.window, opt.eval_batchsize, shuffle=False):
            logits = model(contexts)
            loss = manual_cross_entropy(logits, targets)
            batch_words = targets.numel()
            total_loss += loss.item() * batch_words
            total_words += batch_words

    avg_loss = total_loss / max(total_words, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    print('%s loss: %.4f perplexity: %.2f' % (split_name, avg_loss, ppl))
    return avg_loss, ppl


def save_checkpoint(model, opt, epoch, train_ppl=None, valid_ppl=None, is_best=False):
    if not opt.savename:
        return

    os.makedirs(opt.savename, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.optimizer.state_dict(),
        'vocab': opt.vocab,
        'words': opt.words,
        'config': {
            'threshold': opt.threshold,
            'window': opt.window,
            'd_model': opt.d_model,
            'hidden_dim': opt.hidden_dim,
            'batchsize': opt.batchsize,
            'lr': opt.lr,
        },
        'train_ppl': train_ppl,
        'valid_ppl': valid_ppl,
    }

    path = os.path.join(opt.savename, 'model_epoch_%03d.pt' % epoch)
    torch.save(checkpoint, path)
    torch.save(checkpoint, os.path.join(opt.savename, 'model_latest.pt'))
    if is_best:
        torch.save(checkpoint, os.path.join(opt.savename, 'model_best.pt'))


def write_learning_curve(opt):
    if not opt.savename or not opt.history:
        return

    os.makedirs(opt.savename, exist_ok=True)
    csv_path = os.path.join(opt.savename, 'learning_curve.csv')
    json_path = os.path.join(opt.savename, 'learning_curve.json')
    png_path = os.path.join(opt.savename, 'learning_curve.png')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_ppl', 'valid_loss', 'valid_ppl'])
        writer.writeheader()
        writer.writerows(opt.history)

    with open(json_path, 'w') as f:
        json.dump(opt.history, f, indent=2)

    try:
        import matplotlib.pyplot as plt

        epochs = [row['epoch'] for row in opt.history]
        train_ppl = [row['train_ppl'] for row in opt.history]
        valid_ppl = [row['valid_ppl'] for row in opt.history]

        plt.figure()
        plt.plot(epochs, train_ppl, label='train')
        plt.plot(epochs, valid_ppl, label='valid')
        plt.xlabel('epoch')
        plt.ylabel('perplexity')
        plt.title('Bengio LM learning curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()
    except Exception:
        pass


def train(model,opt):
    train_corpus = torch.tensor(opt.train, dtype=torch.long, device=opt.device)
    valid_corpus = torch.tensor(opt.valid, dtype=torch.long, device=opt.device)
    opt.history = []

    best_valid = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(1, opt.epochs + 1):
        model.train()
        start_time = time.time()
        total_loss = 0.0
        total_words = 0
        num_examples = max(train_corpus.numel() - opt.window, 1)

        for batch_num, (contexts, targets) in enumerate(
            batch_iter(train_corpus, opt.window, opt.batchsize, shuffle=True),
            start=1,
        ):
            opt.optimizer.zero_grad(set_to_none=True)
            logits = model(contexts)
            loss = manual_cross_entropy(logits, targets)
            loss.backward()
            if opt.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_grad)
            opt.optimizer.step()

            batch_words = targets.numel()
            total_loss += loss.item() * batch_words
            total_words += batch_words

            if opt.report_every > 0 and batch_num % opt.report_every == 0:
                elapsed = max(time.time() - start_time, 1e-9)
                percent = 100.0 * total_words / num_examples
                speed = total_words / elapsed
                print(
                    'epoch %d %.1f%% train ppl %.2f words/sec %.0f'
                    % (epoch, percent, math.exp(min(total_loss / total_words, 20.0)), speed)
                )

        train_loss = total_loss / max(total_words, 1)
        train_ppl = math.exp(min(train_loss, 20.0))
        valid_loss, valid_ppl = evaluate(model, valid_corpus, opt, split_name='valid')
        elapsed = max(time.time() - start_time, 1e-9)

        print(
            'epoch %d done train loss %.4f ppl %.2f valid loss %.4f ppl %.2f time %.1fs'
            % (epoch, train_loss, train_ppl, valid_loss, valid_ppl, elapsed)
        )

        opt.history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_ppl': train_ppl,
            'valid_loss': valid_loss,
            'valid_ppl': valid_ppl,
        })
        write_learning_curve(opt)

        improved = valid_ppl < best_valid
        if improved:
            best_valid = valid_ppl
            best_model_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        save_checkpoint(model, opt, epoch, train_ppl, valid_ppl, is_best=improved)

        if opt.patience > 0 and epochs_without_improvement >= opt.patience:
            print('early stopping after %d epochs without validation improvement' % opt.patience)
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print('restored best validation model with perplexity %.2f' % best_valid)

    return

def test_model(model, opt, epoch):
    test_corpus = torch.tensor(opt.test, dtype=torch.long, device=opt.device)
    test_loss, test_ppl = evaluate(model, test_corpus, opt, split_name='test')
    print('final test perplexity: %.2f' % test_ppl)

    if opt.examples:
        print(' ')
        print('example sentence perplexities:')
        for i, example in enumerate(opt.examples, start=1):
            if len(example) <= opt.window:
                print('example %d: not enough tokens for window %d' % (i, opt.window))
                continue
            example_corpus = torch.tensor(example, dtype=torch.long, device=opt.device)
            _, example_ppl = evaluate(model, example_corpus, opt, split_name='example %d' % i)
            print('example %d perplexity: %.2f' % (i, example_ppl))

    return test_loss, test_ppl

def main():
    
    random.seed(10)
    torch.manual_seed(10)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-threshold', type=int, default=3)
    parser.add_argument('-window', type=int, default=5)   
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=100)
    parser.add_argument('-hidden_dim', type=int, default=100)
    parser.add_argument('-batchsize', type=int, default=1024)
    parser.add_argument('-eval_batchsize', type=int, default=4096)
    parser.add_argument('-lr', type=float, default=0.00005)
    parser.add_argument('-patience', type=int, default=5)
    parser.add_argument('-clip_grad', type=float, default=5.0)
    parser.add_argument('-report_every', type=int, default=200)
    parser.add_argument('-savename', type=str)    
    parser.add_argument('-loadname', type=str)    
                
    opt = parser.parse_args()
    opt.verbose = False    
    use_cuda = (not opt.no_cuda) and torch.cuda.is_available()
    opt.device = torch.device('cuda' if use_cuda else 'cpu')
    print('device: %s' % opt.device)
       
    [opt.vocab,opt.words,opt.train] = read_corpus('wiki2.train.txt',[],{},[],opt.threshold)
    print('vocab: %d train: %d' % (len(opt.vocab),len(opt.train)))
    [opt.vocab,opt.words,opt.test] = read_corpus('wiki2.test.txt',opt.vocab,opt.words,[],-1)
    print('vocab: %d test: %d' % (len(opt.vocab),len(opt.test)))
    [opt.vocab,opt.words,opt.valid] = read_corpus('wiki2.valid.txt',opt.vocab,opt.words,[],-1)
    print('vocab: %d test: %d' % (len(opt.vocab),len(opt.valid)))

    print('Train: %7d' % (len(opt.train)))
    print('Test:  %7d' % (len(opt.test)))
    print('Valid: %7d' % (len(opt.valid)))
    print('Vocab: %7d' % (len(opt.vocab)))
    print(' ')
    
    opt.examples = []
    with open('examples.txt','rt') as f:
        for line in f:
            line = line.replace('\n','')
            encoded = encode(line,opt.words)
            text = ''
            for i in range(len(encoded)):
                text = text + opt.vocab[encoded[i]] + ' '
            opt.examples.append(encoded)
            
            print('origianl: %s' % line)
            print('encoded:  %s' % text)
            print(' ')
            
    model = bengio(dim=opt.d_model, 
                   window=opt.window, 
                   batchsize = opt.batchsize, 
                   vocab_size=len(opt.vocab), 
                   hidden_dim=opt.hidden_dim,
                   activation=torch.tanh)
    model = model.to(opt.device)
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)    

    if opt.loadname:
        checkpoint = torch.load(opt.loadname, map_location=opt.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print('loaded weights from %s' % opt.loadname)
    
    train(model,opt)
    test_model(model,opt,-1)
    
if __name__ == "__main__":
    main()     
