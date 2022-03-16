CUDA_VISIBLE_DEVICES="1" \
python -m \
SER.train --bsize 32 --accum 1 --lr 0.001 --MFCC --GE2E \
--triples /content/drive/MyDrive/Speech-Emotion-Recognition/allosaurus+CNN/emodb/total.csv \
--langs "ge" \
--root /content/drive/MyDrive/Speech-Emotion-Recognition/allosaurus+CNN/ --experiment MSMARCO-psg --run msmarco.psg.rronly --maxsteps 20000
