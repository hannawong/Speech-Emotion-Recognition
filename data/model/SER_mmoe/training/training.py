import random
import torch
import numpy as np
from transformers import AdamW
from sklearn.metrics import accuracy_score
from SER_mmoe.utils.amp import MixedPrecisionManager
from SER_mmoe.training.pretrainbatcher import PretrainBatcher
from SER_mmoe.training.batcher_german import PretrainBatcher_ge
from SER_mmoe.training.batcher_persian import PretrainBatcher_pe
from SER_mmoe.training.batcher_french import PretrainBatcher_fr
from SER_mmoe.modeling.ser_model import SER_MODEL
from SER_mmoe.training.utils import split_train_val_test_german,shuf_order
from torch.optim.lr_scheduler import StepLR

DEVICE = "cuda"
print(DEVICE)
LOG = open("log.txt",'w')

langs = ["en","ge","fr"]

def train(args):

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    #EN_TEST_PATH = "./iemocap/_iemocap_04M.test.csv"
    #EN_VAL_PATH = "./iemocap/_iemocap_04M.val.csv"
    #EN_triples = "./iemocap/_iemocap_04M.train.csv"
    PE_triples, PE_TEST_PATH, PE_VAL_PATH = split_train_val_test_german("./persian/total.csv")
    GE_triples, GE_TEST_PATH, GE_VAL_PATH = split_train_val_test_german("./emodb/total.csv")
    FR_triples, FR_TEST_PATH, FR_VAL_PATH = split_train_val_test_german("./french/total.csv")
    EN_triples, EN_TEST_PATH, EN_VAL_PATH = split_train_val_test_german("./iemocap/total.csv")

    best_model = None
    best_uacc = 0.0
    ser_model = SER_MODEL(args,audio_maxlen=200,num_labels=-1)
    ser_model = ser_model.to(DEVICE)

    ge2e_params = list(map(id, ser_model.SpeakerEncoder.parameters()))
    base_params = filter(lambda p: id(p) not in ge2e_params,ser_model.parameters())

    optimizer = AdamW([{'params':base_params},{'params':ser_model.SpeakerEncoder.parameters(),'lr':args.lr*0.01/len(langs)}],lr = args.lr/len(langs),weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)
    optimizer.zero_grad()

    def training(step):
        ser_model.train()
        en_reader = PretrainBatcher(args, EN_triples,(0 if args.rank == -1 else args.rank), args.nranks)
        ge_reader = PretrainBatcher_ge(args,GE_triples,(0 if args.rank == -1 else args.rank), args.nranks)
        pe_reader = PretrainBatcher_pe(args,PE_triples,(0 if args.rank == -1 else args.rank), args.nranks)
        fr_reader = PretrainBatcher_fr(args,FR_triples,(0 if args.rank == -1 else args.rank), args.nranks)
        reader = {"en":en_reader,"ge":ge_reader,"pe":pe_reader,"fr":fr_reader}
        train_loss = 0.0
        start_batch_idx = 0

        i = 0
        accum = 0
        for _ in range(args.maxsteps):
          for lang in shuf_order(langs):
            for batch_idx, BatchSteps in zip(range(start_batch_idx,1), reader[lang]):
                for feat_emb, ge2e_emb, wav2vec_emb, mfcc_emb, labels,length,wav2vec_length,mfcc_length,bloy_embedding,audio_files in BatchSteps: 
                    i += 1 

                    feat_emb = torch.Tensor(feat_emb).to(DEVICE) ##[bz,200,640]
                    ge2e_emb = torch.Tensor(ge2e_emb).to(DEVICE)
                    mfcc_emb = torch.Tensor(mfcc_emb).to(DEVICE) ##[bz,200,24]
                    wav2vec_emb = torch.Tensor(wav2vec_emb).to(DEVICE) ##[bz,200,24]
                    labels = torch.Tensor(labels).to(DEVICE) ##[bz]
                    length = torch.Tensor(length).to(DEVICE)
                    mfcc_length = torch.Tensor(mfcc_length).to(DEVICE)
                    wav2vec_length = torch.Tensor(wav2vec_length).to(DEVICE)
                    bloy_embedding = torch.Tensor(bloy_embedding).to(DEVICE)

                    loss,_ = ser_model(lang,feat_emb,ge2e_emb,mfcc_emb,labels,length,mfcc_length,wav2vec_emb, wav2vec_length,bloy_embedding,audio_files)
                    loss.backward()
                    train_loss += loss.item()
                    accum += 1
                    if accum == args.batch_accum:
                      optimizer.step()
                      optimizer.zero_grad()
                      accum = 0
                    step += 1
            LOG.write("loss: "+str(train_loss/i)+"\n")

    step = 0
    pe_uacc,pe_wacc,en_uacc,en_wacc,ge_uacc,ge_wacc,fr_uacc,fr_wacc = 0,0,0,0,0,0,0,0
    pe_uacc_test,pe_wacc_test,en_uacc_test,en_wacc_test,ge_uacc_test,ge_wacc_test,fr_uacc_test,fr_wacc_test = 0,0,0,0,0,0,0,0
    for epoch in range(20):
        scheduler.step()
        print('Epoch:', epoch,'LR:', scheduler.get_lr())
        LOG.write("="*30+"epoch: "+str(epoch)+"="*30+">"+"\n")
        training(step)
        if "ge" in langs:
            ge_uacc,ge_wacc = evaluate(args,ser_model,GE_VAL_PATH,7,"ge")
        if "pe" in langs:
            pe_uacc,pe_wacc = evaluate(args,ser_model,PE_VAL_PATH,6,"pe")
        if "en" in langs:
            en_uacc,en_wacc = evaluate(args,ser_model,EN_VAL_PATH,4,"en")
        if "fr" in langs:
            fr_uacc,fr_wacc = evaluate(args,ser_model,FR_VAL_PATH,7,"fr")

        print("best validation score now is ",best_uacc)
        if "ge" in langs:
            ge_uacc_test,ge_wacc_test = evaluate(args,ser_model,GE_TEST_PATH,7,"ge")
        if "en" in langs:
            en_uacc_test,en_wacc_test = evaluate(args,ser_model,EN_TEST_PATH,4,"en")
        if "pe" in langs:
            pe_uacc_test,pe_wacc_test = evaluate(args,ser_model,PE_TEST_PATH,6,"pe")
        if "fr" in langs:
            fr_uacc_test,fr_wacc_test = evaluate(args,ser_model,FR_TEST_PATH,7,"fr")
        print("On test set german",ge_uacc_test," ,",ge_wacc_test)
        print("On test set english",en_uacc_test," ,",en_wacc_test)
        print("On test set persian",pe_uacc_test," ,",pe_wacc_test)
        print("On test set french",fr_uacc_test," ,",fr_wacc_test)

        if ge_uacc+pe_uacc+en_uacc+fr_uacc >= best_uacc:
            print("saving best model...")
            best_uacc = ge_uacc+pe_uacc+en_uacc+fr_uacc
            torch.save(ser_model, 'ser.pt')
    
    print("****finish training, now test on testset*****")
    best_model = torch.load('ser.pt')
    if "ge" in langs:
        evaluate(args,best_model,GE_TEST_PATH,7,"ge")
    if "pe" in langs:
        evaluate(args,best_model,PE_TEST_PATH,6,"pe")
    if "en" in langs:
        evaluate(args,best_model,EN_TEST_PATH,4,"en")
    if "fr" in langs:
        evaluate(args,best_model,FR_TEST_PATH,7,"fr")

def evaluate(args,model,path,CLASS_NUM,langs):
    print("begin evaluate...",langs)
    if langs == "en":
        reader = PretrainBatcher(args, path,(0 if args.rank == -1 else args.rank), args.nranks)
    if langs == "ge":
        reader = PretrainBatcher_ge(args, path,(0 if args.rank == -1 else args.rank), args.nranks)
    if langs == "pe":
        reader = PretrainBatcher_pe(args, path,(0 if args.rank == -1 else args.rank), args.nranks)
    if langs == "fr":
        reader = PretrainBatcher_fr(args, path,(0 if args.rank == -1 else args.rank), args.nranks)
    start_batch_idx = 0
    class_tot = [0]*CLASS_NUM 
    class_correct = [0]*CLASS_NUM
    i = 0
    tot_acc = 0

    with torch.no_grad():
        for batch_idx, BatchSteps in zip(range(start_batch_idx,args.maxsteps), reader):
            for feat_emb,ge2e_emb,wav2vec_emb, mfcc_emb,labels ,length,wav2vec_length,mfcc_length,bloy_embedding,audio_files in BatchSteps: 
                i += 1 
                feat_emb = torch.Tensor(feat_emb).to(DEVICE) ##[bz,100,230]
                labels = torch.Tensor(labels).to(DEVICE) ##[bz]
                ge2e_emb = torch.Tensor(ge2e_emb).to(DEVICE) ##[bz,100,230]
                length = torch.Tensor(length).to(DEVICE)
                mfcc_emb = torch.Tensor(mfcc_emb).to(DEVICE)
                mfcc_length = torch.Tensor(mfcc_length).to(DEVICE)
                wav2vec_emb = torch.Tensor(wav2vec_emb).to(DEVICE) ##[bz,100,230]
                wav2vec_length = torch.Tensor(wav2vec_length).to(DEVICE)
                bloy_embedding = torch.Tensor(bloy_embedding).to(DEVICE)
                loss, output = model(langs,feat_emb,ge2e_emb,mfcc_emb,labels,length,mfcc_length,wav2vec_emb,wav2vec_length,bloy_embedding,audio_files)
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