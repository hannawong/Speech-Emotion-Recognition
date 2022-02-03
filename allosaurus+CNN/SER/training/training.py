import random
import torch
import numpy as np
from transformers import AdamW

from SER.utils.amp import MixedPrecisionManager
from SER.training.pretrainbatcher import PretrainBatcher
from SER.parameters import DEVICE
from SER.modeling.ser_model import SER_MODEL
from SER.utils.utils import print_message
from SER.training.utils import manage_checkpoints

LOG = open("log.txt",'w')
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

    ser_model = SER_MODEL(audio_maxlen=100,dim=128)

    if args.checkpoint is not None:
        assert args.resume_optimizer is False
        print_message(f"#> Starting from checkpoint {args.checkpoint} -- but NOT the optimizer!")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        try:
            ser_model.load_state_dict(checkpoint['model_state_dict'])
        except:
            print_message("[WARNING] Loading checkpoint with strict=False")
            ser_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if args.rank == 0:
        torch.distributed.barrier()

    ser_model = ser_model.to(DEVICE)
    ser_model.train()

    if args.distributed:
        ser_model = torch.nn.parallel.DistributedDataParallel(ser_model, device_ids=[args.rank],
                                                            output_device=args.rank,
                                                            find_unused_parameters=True)


    amp = MixedPrecisionManager(args.amp)
    optimizer = AdamW(filter(lambda p: p.requires_grad, ser_model.parameters()))

    def training(step):
        reader = PretrainBatcher(args, args.triples,(0 if args.rank == -1 else args.rank), args.nranks)
        train_loss = 0.0
        start_batch_idx = 0
        optimizer.zero_grad()

        if args.resume:
            assert args.checkpoint is not None
            start_batch_idx = checkpoint['batch']

            reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

        for batch_idx, BatchSteps in zip(range(start_batch_idx,args.maxsteps), reader):
            for feat_emb, labels in BatchSteps:    
                feat_emb = torch.Tensor(feat_emb).cuda() ##[32,100,230]
                labels = torch.Tensor(labels).cuda() ##[32]
                with amp.context():
                    loss = ser_model(feat_emb,labels)
                    amp.backward(loss)
                    train_loss += loss.item()

                avg_loss = train_loss / (batch_idx+1)
                msg = print_message(step, avg_loss)
                LOG.write(msg+"\n")
                amp.step(ser_model, optimizer)
                step += 1
                

    step = 0
    for epoch in range(10):
        training(step)
        manage_checkpoints(args, ser_model, optimizer, step+1)
