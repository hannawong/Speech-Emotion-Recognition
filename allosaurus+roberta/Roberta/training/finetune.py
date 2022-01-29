import random
import torch
import numpy as np
from transformers import AdamW

from Roberta.utils.amp import MixedPrecisionManager
from Roberta.training.finetunebatcher import FinetuneBatcher
from Roberta.parameters import DEVICE
from Roberta.modeling.colbert import ColBERT
from Roberta.utils.utils import print_message
from Roberta.training.utils import get_mask, manage_checkpoints


def train(args):

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    if args.distributed:
        torch.cuda.manual_seed_all(12345)
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)
    
    
    if args.rank not in [-1, 0]:
        torch.distributed.barrier()

    colbert = ColBERT.from_pretrained('roberta-base',
                                            query_maxlen=32,
                                            dim=128,
                                            mask_punctuation=True)

    if args.checkpoint is not None:
        assert args.resume_optimizer is False
        print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        try:
            colbert.load_state_dict(checkpoint['model_state_dict'])
        except:
            print_message("[WARNING] Loading checkpoint with strict=False")
            colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if args.rank == 0:
        torch.distributed.barrier()

    colbert = colbert.to(DEVICE)
    colbert.train()

    if args.distributed:
        colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[args.rank],
                                                            output_device=args.rank,
                                                            find_unused_parameters=True)


    amp = MixedPrecisionManager(args.amp)
    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=args.lr, eps=1e-8)

    def training(step):
        reader = FinetuneBatcher(args, args.triples,(0 if args.rank == -1 else args.rank), args.nranks)
        train_loss_mlm = 0.0
        start_batch_idx = 0
        optimizer.zero_grad()

        if args.resume:
            assert args.checkpoint is not None
            start_batch_idx = checkpoint['batch']

            reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

        for batch_idx, BatchSteps in zip(range(start_batch_idx,args.maxsteps), reader):
            for queries in BatchSteps:
                sentence = queries[0]
                sentence_mask = queries[1]
                labels = queries[2]
                print(sentence)
                print(sentence_mask)
                print(labels)
                  
                with amp.context():
                    #queries_mlm,labels_mlm = get_mask(queries,args,reader)
                    loss = colbert((sentence,sentence_mask),"ser",labels)
                    print(loss)

                    amp.backward(loss)
                    train_loss_mlm += loss.item()

                avg_loss = train_loss_mlm / (batch_idx+1)
                print_message(step, avg_loss)
                amp.step(colbert, optimizer)
                step += 1
                

    step = 0
    for epoch in range(10):
        training(step)
        manage_checkpoints(args, colbert, optimizer, step+1)
