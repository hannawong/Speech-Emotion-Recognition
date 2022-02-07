import random
from time import sleep
import torch
import numpy as np
from transformers import AdamW

from SER.utils.amp import MixedPrecisionManager
from SER.training.pretrainbatcher import PretrainBatcher
from SER.parameters import DEVICE
from SER.modeling.ser_model import SER_MODEL
from SER.utils.utils import print_message
from SER.training.utils import manage_checkpoints

LOG = open("log_.txt",'w')
CLASS_NUM = 4
TEST_PATH = "/data/jiayu_xiao/project/wzh/Speech_Emotion_Recognition/allosaurus+CNN/iemocap/_iemocap_03M.test.csv"
VAL_PATH = "/data/jiayu_xiao/project/wzh/Speech_Emotion_Recognition/allosaurus+CNN/iemocap/_iemocap_03M.val.csv"

def train(args):

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    ser_model = SER_MODEL(audio_maxlen=100)

    best_model = None
    best_uacc = 0.0
    best_wacc = 0.0

    if args.distributed:
        torch.cuda.manual_seed_all(12345)
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)
    
    
    if args.rank not in [-1, 0]:
        torch.distributed.barrier()

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

    if args.distributed:
        ser_model = torch.nn.parallel.DistributedDataParallel(ser_model, device_ids=[args.rank],
                                                            output_device=args.rank,
                                                            find_unused_parameters=True)


    amp = MixedPrecisionManager(args.amp)
    optimizer = AdamW(filter(lambda p: p.requires_grad, ser_model.parameters()),lr = 0.005,weight_decay=1e-5)

    def training(step):
        ser_model.train()
        reader = PretrainBatcher(args, args.triples,(0 if args.rank == -1 else args.rank), args.nranks)
        train_loss = 0.0
        start_batch_idx = 0

        if args.resume:
            assert args.checkpoint is not None
            start_batch_idx = checkpoint['batch']

            reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

        i = 0
        for batch_idx, BatchSteps in zip(range(start_batch_idx,args.maxsteps), reader):
            for feat_emb, labels in BatchSteps: 
                i += 1 
                optimizer.zero_grad()
                feat_emb = torch.Tensor(feat_emb).cuda() ##[32,100,230]
                labels = torch.Tensor(labels).cuda() ##[32]
                with amp.context():
                    loss,_ = ser_model(feat_emb,labels)
                    amp.backward(loss)
                    print(loss)
                    train_loss += loss.item()

                avg_loss = train_loss / (batch_idx+1)
                msg = print_message(step, avg_loss)
                amp.step(ser_model, optimizer)
                step += 1
        LOG.write("loss: "+str(train_loss/i)+"\n")

            
    step = 0
    for epoch in range(60):
        print("="*30+"epoch: "+str(epoch)+"="*30+">")
        LOG.write("="*30+"epoch: "+str(epoch)+"="*30+">"+"\n")
        training(step)
        uacc,wacc = evaluate(args,ser_model,TEST_PATH)
        if not best_model:
            best_model = ser_model
        else:
            if uacc > best_uacc:
                print("saving best model...")
                best_uacc = uacc
                torch.save(ser_model,"checkpoint.dnn")
        #manage_checkpoints(args, ser_model, optimizer, step+1)
    
    print("finish training, now test on testset")
    best_model = torch.load("checkpoint.dnn")
    evaluate(args,best_model,TEST_PATH)

from sklearn.metrics import accuracy_score
def evaluate(args,model,path):
    reader = PretrainBatcher(args, path,(0 if args.rank == -1 else args.rank), args.nranks)
    start_batch_idx = 0
    class_tot = [0]*CLASS_NUM 
    class_correct = [0]*CLASS_NUM
    i = 0
    tot_acc = 0

    with torch.no_grad():
        for batch_idx, BatchSteps in zip(range(start_batch_idx,args.maxsteps), reader):
            for feat_emb, labels in BatchSteps: 
                i += 1 
                feat_emb = torch.Tensor(feat_emb).cuda() ##[32,100,230]
                labels = torch.Tensor(labels).cuda() ##[32]
                loss, output = model(feat_emb,labels)
                pred_label = torch.argmax(output,dim = 1)
                acc = accuracy_score(pred_label.cpu().detach(),labels.cpu().detach())
                tot_acc += acc
                ####### compute unweighted accuracy  ########
                pred_label = pred_label.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                for j in range(len(pred_label)):
                        class_tot[int(labels[j])] += 1
                        if int(pred_label[j]) == int(labels[j]):
                            class_correct[int(labels[j])] += 1
    unweight_acc = []
    for j in range(CLASS_NUM):
        unweight_acc.append(class_correct[j]/class_tot[j])
    
    LOG.write("unweighted test accuracy"+str(sum(unweight_acc)/4)+"\n")
    print("UA= ",sum(unweight_acc)/4)              
    LOG.write("weighted test accuracy"+str(tot_acc/i)+"\n")
    print("weighted test accuracy"+str(tot_acc/i)+"\n")
    return sum(unweight_acc)/4 , tot_acc/i