import random
from time import sleep
import torch
import numpy as np
from transformers import AdamW
from sklearn.metrics import accuracy_score
from SER.utils.amp import MixedPrecisionManager
from SER.training.pretrainbatcher import PretrainBatcher
from SER.modeling.ser_model import SER_MODEL
from SER.utils.utils import print_message

DEVICE = torch.device("cuda")
LOG = open("log.txt",'w')
CLASS_NUM = 4
TEST_PATH = "/data1/jiayu_xiao/project/wzh_recommendation/Speech-Emotion-Recognition/allosaurus+CNN/iemocap/_iemocap_05M.test.csv"
VAL_PATH = "/data1/jiayu_xiao/project/wzh_recommendation/Speech-Emotion-Recognition/allosaurus+CNN/iemocap/_iemocap_05M.test.csv"

def train(args):

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    ser_model = SER_MODEL(audio_maxlen=100)

    best_model = None
    best_uacc = 0.0
    best_wacc = 0.0
    ser_model = ser_model.cuda()

    amp = MixedPrecisionManager(args.amp)
    optimizer = AdamW(filter(lambda p: p.requires_grad, ser_model.parameters()),lr = 0.0005,weight_decay=1e-5)

    def training(step):
        ser_model.train()
        reader = PretrainBatcher(args, args.triples,(0 if args.rank == -1 else args.rank), args.nranks)
        train_loss = 0.0
        start_batch_idx = 0

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
    sleep(2)

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
    weight_acc = []
    for j in range(CLASS_NUM):
        unweight_acc.append(class_correct[j]/(class_tot[j]+0.0001))
        weight_acc.append((class_correct[j]/class_tot[j])*(class_tot[j]/sum(class_tot)))
    LOG.write("unweighted test accuracy"+str(sum(unweight_acc)/4)+"\n")
    print("UA= ",sum(unweight_acc)/4)              
    LOG.write("weighted test accuracy"+str(tot_acc/(i+0.0001))+"\n")
    print("weighted test accuracy"+str(tot_acc/(i+0.0001))+"\n")
    return sum(unweight_acc)/4 , tot_acc/i