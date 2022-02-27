CUDA_VISIBLE_DEVICES="1" \
python -m \
SER.train --bsize 32 --accum 1 --lr 0.001 --GE2E --MFCC \
--triples /data1/jiayu_xiao/project/wzh_recommendation/Speech-Emotion-Recognition/allosaurus+CNN/emodb/total.csv \
--langs "ge" \
--root /data1/jiayu_xiao/project/wzh_recommendation/Speech-Emotion-Recognition/allosaurus+CNN --experiment MSMARCO-psg --run msmarco.psg.rronly --maxsteps 20000