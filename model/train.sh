CUDA_VISIBLE_DEVICES="1" \
python -m SER_mmoe.train --bsize 32 --accum 1 --batch_accum 1 --lr 0.0005 --GE2E --MFCC \
--triples allosaurus+CNN/emodb/total.csv \
--langs "ge" \
--root /data/zihan_wang/project/wzh/CL/Speech-Emotion-Recognition/model --experiment MSMARCO-psg --run msmarco.psg.rronly --maxsteps 20000
