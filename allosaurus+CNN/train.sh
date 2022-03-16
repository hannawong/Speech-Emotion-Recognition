CUDA_VISIBLE_DEVICES="0" \
python -m \
SER.train --bsize 512 --accum 1 --lr 0.001 --GE2E \
--triples /content/drive/MyDrive/Speech-Emotion-Recognition/allosaurus+CNN/iemocap/_iemocap_03F.train.csv \
--langs "en" \
--root /content/drive/MyDrive/Speech-Emotion-Recognition/allosaurus+CNN/ --experiment MSMARCO-psg --run msmarco.psg.rronly --maxsteps 20000