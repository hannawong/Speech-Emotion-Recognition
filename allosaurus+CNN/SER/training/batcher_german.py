from functools import partial
from SER.training.utils import  tensorize_triples

class PretrainBatcher_ge():
    def __init__(self, args, path, rank=0, nranks=1):
        self.args = args
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
        cls_label_map = {"bored":0, "fear":1, "angry":2, "happy":3,"disgust":4,"sad":5}
        audio_filenames = []
        labels = []
        line_idx = 0
        for line_idx, line in zip(range(self.bsize * self.nranks), self.reader):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue
            line_split = line.strip().split(",")
            audio_file = line_split[2]
            label = cls_label_map[line_split[1]]
            labels.append(label)
            audio_filenames.append(audio_file)

        self.position += line_idx + 1

        if len(audio_filenames) < self.bsize:
            raise StopIteration

        return self.collate(audio_filenames,labels)

    def collate(self, audio_filenames,labels):
        assert len(audio_filenames) == self.bsize
        return self.tensorize_triples(self.args,audio_filenames, labels, self.bsize // self.accumsteps)


    def skip_to_batch(self, batch_idx, intended_batch_size):
        self._reset_triples()

        print(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')

        _ = [self.reader.readline() for _ in range(batch_idx * intended_batch_size)]

        return None
