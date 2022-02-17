from allosaurus.app import read_recognizer
from allosaurus.model import get_all_models, resolve_model_name
from allosaurus.bin.download_model import download_model
from pathlib import Path
import pickle
import argparse

TRAIN_PATH = "/data1/jiayu_xiao/project/wzh_recommendation/Speech-Emotion-Recognition/allosaurus+CNN/emodb/total.csv"
ALLO_EMB_PATH = "/data1/jiayu_xiao/project/wzh/data/ger_allo_embedding/"

def get_allosaurus_embedding(input):
    # input file/path
    input_path = Path(input)
    # check file format
    assert input.endswith('.wav'), " Error: Please use a wav file. other audio files can be converted to wav by sox"
    # run inference
    phones = recognizer.recognize(input, args.lang, args.topk, args.emit, args.timestamp)
    return phones


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Allosaurus phone recognizer')
    parser.add_argument('-d', '--device_id', type=int, default=-1, help='specify cuda device id to use, -1 means no cuda and will use cpu for inference')
    parser.add_argument('-m', '--model', type=str, default='latest', help='specify which model to use. default is to use the latest local model')
    parser.add_argument('-l', '--lang', type=str,  default='ipa',help='specify which language inventory to use for recognition. default is to use all phone inventory')
    #parser.add_argument('-i', '--input', type=str, required=True, help='specify your input wav file/directory')
    parser.add_argument('-o', '--output', type=str, default='stdout', help='specify output file. the default will be stdout')
    parser.add_argument('-k', '--topk', type=int, default=1, help='output k phone for each emitting frame')
    parser.add_argument('-t', '--timestamp', type=bool, default=False, help='attach *approximate* timestamp for each phone, note that the timestamp might not be accurate')
    parser.add_argument('-p', '--prior', type=str, required=False, default=None, help='supply prior to adjust phone predictions')
    parser.add_argument('-e', '--emit', type=float, required=False, default=1.0, help='specify how many phones to emit. A larger number can emit more phones and a smaller number would suppress emission, default is 1.0')
    parser.add_argument('-a', '--approximate', type=bool, default=False, help='the phone inventory can still hardly to cover all phones. You can use turn on this flag to map missing phones to other similar phones to recognize. The similarity is measured with phonological features')

    args = parser.parse_args()

    # download specified model automatically if no model exists
    if len(get_all_models()) == 0:
        download_model('latest')

    # resolve model's name
    model_name = resolve_model_name(args.model)
    if model_name == "none":
        print("Model ", model_name, " does not exist. Please download this model or use an existing model in list_model")
        exit(0)

    args.model = model_name

    # create recognizer
    recognizer = read_recognizer(args)

    # output file descriptor
    output_fd = None
    if args.output != 'stdout':
        output_fd = open(args.output, 'w', encoding='utf-8')

    INPUT_FILE = open(TRAIN_PATH,"r").read().split("\n")
    for i in range(len(INPUT_FILE)):
        print(i)
        line = INPUT_FILE[i]
        line = line.split(",")
        file_name = line[1]
        allo_emb = get_allosaurus_embedding(file_name)  ##(1,65,230)
        output = open(ALLO_EMB_PATH+file_name.split("/")[-1]+'.pkl', 'wb')
        pickle.dump(allo_emb, output)