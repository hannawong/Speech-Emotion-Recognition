CUDA_VISIBLE_DEVICES="0" \
python -m \
SER.train --bsize 32 --accum 1 \
--triples /data/jiayu_xiao/project/wzh/Speech_Emotion_Recognition/allosaurus+CNN/iemocap/_iemocap_01F.train.csv \
--langs "en" \
--root /data/jiayu_xiao/project/wzh/Speech_Emotion_Recognition/allosaurus+CNN --experiment MSMARCO-psg --run msmarco.psg.rronly --maxsteps 20000