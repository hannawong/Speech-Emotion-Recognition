from functools import partial
from SER.training.utils import  tensorize_triples

class PretrainBatcher():
    def __init__(self, args, path, rank=0, nranks=1):
        self.rank, self.nranks = rank, nranks
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.tensorize_triples = partial(tensorize_triples)
        self.triples_path = path
        self._reset_triples()

    def _reset_triples(self):
        self.reader = open(self.triples_path, mode='r', encoding="utf-8")
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        cls_label_map = {"e0":0, "e1":1, "e2":2, "e3":3}
        audio_filenames = []
        labels = []
        line_idx = 0
        for line_idx, line in zip(range(self.bsize * self.nranks), self.reader):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue
            line_split = line.strip().split(",")
            audio_file = line_split[0]
            label = cls_label_map[line_split[1]]
            labels.append(label)
            audio_filenames.append(audio_file)

        self.position += line_idx + 1

        if len(audio_filenames) < self.bsize:
            raise StopIteration

        return self.collate(audio_filenames,labels)

    def collate(self, audio_filenames,labels):
        assert len(audio_filenames) == self.bsize
        return self.tensorize_triples(audio_filenames, labels, self.bsize // self.accumsteps)


    def skip_to_batch(self, batch_idx, intended_batch_size):
        self._reset_triples()

        print(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')

        _ = [self.reader.readline() for _ in range(batch_idx * intended_batch_size)]

        return None
