import os, spacy
from torchtext import data
import pandas as pd

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class tokenize(object):
    def __init__(self, lang):
        self.nlp = spacy.load(lang)

    def tokenizer(self, sentence):
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]


def read_data(dataset, mode, src_field, trg_field):
    src_data = dataset[mode][src_field]
    trg_data = dataset[mode][trg_field]
    return src_data, trg_data


def create_fields(lang):
    print("loading spacy tokenizers...")

    tokenizer = tokenize(lang)
    tokenizer = data.Field(lower=False, tokenize=tokenizer.tokenizer, init_token='<sos>', eos_token='<eos>')

    return tokenizer


def create_dataset(src_data, trg_data, batchsize, device, tokenizer, istrain=True):
    print("creating dataset and iterator... ")
    raw_data = {'src': [line for line in src_data], 'trg': [line for line in trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    df.to_csv("translate_transformer_temp.csv", index=False)

    vocab = {'vocab': [line for line in src_data + trg_data]}
    vocab = pd.DataFrame(vocab, columns=["vocab"])
    vocab.to_csv("vocab.csv", index=False)
    vocab = data.TabularDataset('./vocab.csv', format='csv', fields=[('vocab', tokenizer)])

    data_fields = [('src', tokenizer), ('trg', tokenizer)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=batchsize, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=istrain, shuffle=True)
    os.remove('translate_transformer_temp.csv')
    os.remove('vocab.csv')

    if istrain:
        tokenizer.build_vocab(vocab)

    return train_iter