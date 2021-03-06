import random
import torch
import numpy as np
from transformers import AdamW
from sklearn.metrics import accuracy_score
from SER.utils.amp import MixedPrecisionManager
from SER.training.pretrainbatcher import PretrainBatcher
from SER.training.batcher_german import PretrainBatcher_ge
from SER.training.batcher_persian import PretrainBatcher_pe
from SER.modeling.ser_model import SER_MODEL
from SER.utils.utils import print_message
from SER.training.utils import split_train_val_test_german
from torch.optim.lr_scheduler import StepLR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
LOG = open("log.txt",'w')
CLASS_NUM = 4

def train(args):

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    if args.langs == "en":
        ser_model = SER_MODEL(args,audio_maxlen=200,num_labels=4)
    if args.langs == "ge":
        ser_model = SER_MODEL(args,audio_maxlen=200,num_labels=6)
        global CLASS_NUM
        CLASS_NUM = 6
    if args.langs == "pe":
        ser_model = SER_MODEL(args,audio_maxlen=200,num_labels=6)
        CLASS_NUM = 6


    TEST_PATH = args.triples[:-10]+".test.csv"
    VAL_PATH = args.triples[:-10]+".val.csv"

    if args.langs == "ge" or args.langs == "pe":
        args.triples, TEST_PATH, VAL_PATH = split_train_val_test_german(args.triples)

    best_model = None
    best_uacc = 0.0
    best_wacc = 0.0
    ser_model = ser_model.to(DEVICE)

    amp = MixedPrecisionManager(args.amp)

    ge2e_params = list(map(id, ser_model.SpeakerEncoder.parameters()))
    base_params = filter(lambda p: id(p) not in ge2e_params,ser_model.parameters())

    optimizer = AdamW([{'params':base_params},{'params':ser_model.SpeakerEncoder.parameters(),'lr':args.lr*0.1}],lr = args.lr,weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)
    optimizer.zero_grad()

    def training(step):
        ser_model.train()
        if args.langs == "en":
            reader = PretrainBatcher(args, args.triples,(0 if args.rank == -1 else args.rank), args.nranks)
        if args.langs == "ge":
            reader = PretrainBatcher_ge(args,args.triples,(0 if args.rank == -1 else args.rank), args.nranks)
        if args.langs == "pe":
            reader = PretrainBatcher_pe(args,args.triples,(0 if args.rank == -1 else args.rank), args.nranks)
        train_loss = 0.0
        start_batch_idx = 0

        i = 0
        accum = 0
        for batch_idx, BatchSteps in zip(range(start_batch_idx,args.maxsteps), reader):
            for feat_emb, ge2e_emb, mfcc_emb, labels,length,mfcc_length,audio_files in BatchSteps: 
                i += 1 
                feat_emb = torch.Tensor(feat_emb).to(DEVICE) ##[bz,200,640]
                ge2e_emb = torch.Tensor(ge2e_emb).to(DEVICE)
                mfcc_emb = torch.Tensor(mfcc_emb).to(DEVICE) ##[bz,200,24]
                labels = torch.Tensor(labels).to(DEVICE) ##[bz]
                length = torch.Tensor(length).to(DEVICE)
                mfcc_length = torch.Tensor(mfcc_length).to(DEVICE)
                

                loss,_ = ser_model(feat_emb,ge2e_emb,mfcc_emb,labels,length,mfcc_length,audio_files)
                loss.backward()
                print(loss)
                train_loss += loss.item()
                accum += 1
                if accum == args.batch_accum:
                  optimizer.step()
                  optimizer.zero_grad()
                  accum = 0

                avg_loss = train_loss / (batch_idx+1)
                msg = print_message(step, avg_loss)
                step += 1
        LOG.write("loss: "+str(train_loss/i)+"\n")

            
    step = 0
    for epoch in range(20):
        scheduler.step()
        print('Epoch:', epoch,'LR:', scheduler.get_lr())
        LOG.write("="*30+"epoch: "+str(epoch)+"="*30+">"+"\n")
        training(step)
        uacc,wacc = evaluate(args,ser_model,VAL_PATH)
        print("best validation score now is ",best_uacc)
        uacc_test,wacc_test = evaluate(args,ser_model,TEST_PATH)
        print("On test set",uacc_test," ,",wacc_test)
        if not best_model:
            best_model = ser_model
        else:
            if uacc >= best_uacc:
                print("saving best model...")
                best_uacc = uacc
                best_model = ser_model
                torch.save(best_model,"checkpoint.dnn")
    
    print("finish training, now test on testset")
    #best_model = torch.load("checkpoint.dnn")
    evaluate(args,best_model,TEST_PATH)

def evaluate(args,model,path):
    print(CLASS_NUM,"=========")
    print("begin evaluate...")
    if args.langs == "en":
        reader = PretrainBatcher(args, path,(0 if args.rank == -1 else args.rank), args.nranks)
    if args.langs == "ge":
        reader = PretrainBatcher_ge(args, path,(0 if args.rank == -1 else args.rank), args.nranks)

    if args.langs == "pe":
        reader = PretrainBatcher_pe(args, path,(0 if args.rank == -1 else args.rank), args.nranks)
    start_batch_idx = 0
    class_tot = [0]*CLASS_NUM 
    class_correct = [0]*CLASS_NUM
    i = 0
    tot_acc = 0

    with torch.no_grad():
        for batch_idx, BatchSteps in zip(range(start_batch_idx,args.maxsteps), reader):
            for feat_emb,ge2e_emb, mfcc_emb,labels ,length,mfcc_length,audio_files in BatchSteps: 
                i += 1 
                feat_emb = torch.Tensor(feat_emb).to(DEVICE) ##[bz,100,230]
                labels = torch.Tensor(labels).to(DEVICE) ##[bz]
                ge2e_emb = torch.Tensor(ge2e_emb).to(DEVICE) ##[bz,100,230]
                length = torch.Tensor(length).to(DEVICE)
                mfcc_emb = torch.Tensor(mfcc_emb).to(DEVICE)
                mfcc_length = torch.Tensor(mfcc_length).to(DEVICE)
                

                loss, output = model(feat_emb,ge2e_emb,mfcc_emb,labels,length,mfcc_length,audio_files)
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
    weight_acc = []
    for j in range(CLASS_NUM):
        unweight_acc.append(class_correct[j]/(class_tot[j]+0.0001))
        weight_acc.append((class_correct[j]/(class_tot[j]+0.0001))*(class_tot[j]/(sum(class_tot)+0.0001)))
    LOG.write("unweighted test accuracy"+str(sum(unweight_acc)/CLASS_NUM)+"\n")
    print("UA= ",sum(unweight_acc)/CLASS_NUM)              
    LOG.write("weighted test accuracy"+str(tot_acc/(i+0.0001))+"\n")
    print("weighted test accuracy"+str(tot_acc/(i+0.0001))+"\n")
    return sum(unweight_acc)/CLASS_NUM , tot_acc/(i+0.0001)