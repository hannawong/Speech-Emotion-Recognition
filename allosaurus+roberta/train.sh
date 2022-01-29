CUDA_VISIBLE_DEVICES="0" \
python -m \
colXLM.train --mask-punctuation --bsize 4 --accum 1 --mlm_probability 0.15 \
--triples /data/jiayu_xiao/my_data/Dataset/triples.train.en.tsv \
--langs "en" \
--root /data/jiayu_xiao/project/wzh/Speech_Emotion_Recognition/allosaurus+roberta --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.rronly --maxsteps 20000