CUDA_VISIBLE_DEVICES="1" \
python -m \
SER.train --bsize 128 --accum 1 --lr 0.0005 --GE2E \
--triples /data1/jiayu_xiao/project/wzh_recommendation/Speech-Emotion-Recognition/allosaurus+CNN/iemocap/_iemocap_01M.train.csv \
--langs "en" \
--root /data1/jiayu_xiao/project/wzh_recommendation/Speech-Emotion-Recognition/allosaurus+CNN --experiment MSMARCO-psg --run msmarco.psg.rronly --maxsteps 20000