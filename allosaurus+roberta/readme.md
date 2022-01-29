# Allosaurus + Roberta

### Pretrain Roberta with MLM task
Firstly, replace `--triples` in `train.sh` with the path to your pretraining corpus (phones extracted by Allosaurus). 
Then run `sh train.sh`

### Finetune on SER dataset
Replace `--triples` in `finetune.sh` with the path to your SER dataset (with phones extracted by ALlosaurus)
Run `sh finetune.sh`