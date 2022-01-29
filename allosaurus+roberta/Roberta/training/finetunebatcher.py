
from functools import partial
import torch
from Roberta.modeling.tokenization import QueryTokenizer, tensorize_triples_classification


class FinetuneBatcher():
    def __init__(self, args, path, rank=0, nranks=1):
        self.rank, self.nranks = rank, nranks
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.tensorize_triples = partial(tensorize_triples_classification, self.query_tokenizer)

        self.triples_path = path
        self._reset_triples()

    def _reset_triples(self):
        self.reader = open(self.triples_path, mode='r', encoding="utf-8")
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        sentences = []
        labels = []
        line_idx = 0
        for line_idx, line in zip(range(self.bsize * self.nranks), self.reader):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue

            line_list = line.strip().split("\t")
            sentence = line_list[3]
            label = int(line_list[1])
            labels.append(label)
            sentences.append(sentence)

        self.position += line_idx + 1

        if len(sentences) < self.bsize:
            raise StopIteration

        return self.collate(sentences,labels)

    def collate(self, sentences,labels):
        assert len(sentences) == self.bsize
        return self.tensorize_triples(sentences, labels, self.bsize // self.accumsteps)


    def skip_to_batch(self, batch_idx, intended_batch_size):
        self._reset_triples()

        print(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')

        _ = [self.reader.readline() for _ in range(batch_idx * intended_batch_size)]

        return None
