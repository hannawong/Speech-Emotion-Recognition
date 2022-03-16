CUDA_VISIBLE_DEVICES="1" \
python -m \
SER.train --bsize 256 --accum 1 --lr 0.001 --MFCC --GE2E \
--triples /data1/jiayu_xiao/project/wzh_recommendation/Speech-Emotion-Recognition/allosaurus+CNN/persian/total.csv \
--langs "pe" \
--root /data1/jiayu_xiao/project/wzh_recommendation/Speech-Emotion-Recognition/allosaurus+CNN --experiment MSMARCO-psg --run msmarco.psg.rronly --maxsteps 20000
