CUDA_VISIBLE_DEVICES="0" \
python -m \
Roberta.finetune --mask-punctuation --bsize 4 --accum 1 --mlm_probability 0.15 \
--triples /data/jiayu_xiao/project/wzh/Roberta/cola_public/raw/in_domain_train.tsv \
--langs "en" \
--root /data/jiayu_xiao/project/wzh/Speech_Emotion_Recognition/allosaurus+roberta --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.rronly --maxsteps 20000