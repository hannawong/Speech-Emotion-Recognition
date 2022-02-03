###   Get training set and testset  ###


PATH = '/data/jiayu_xiao/project/wzh/Speech_Emotion_Recognition/allosaurus+CNN/iemocap/'
TRAIN_PATH = "/data/jiayu_xiao/project/wzh/Speech_Emotion_Recognition/allosaurus+CNN/iemocap/train.csv"
TEST_PATH =  "/data/jiayu_xiao/project/wzh/Speech_Emotion_Recognition/allosaurus+CNN/iemocap/test.csv"

OUT_TRAIN = open(TRAIN_PATH,'w')
OUT_TEST = open(TEST_PATH,'w')

ids = [1,2,3,4,5]
fm = ['M','F']

test_id = 5

for i in ids:
    for f in fm:
        filename_train = PATH+"_iemocap_0"+str(i)+f+".train.csv"
        filename_test = PATH+"_iemocap_0"+str(i)+f+".test.csv"
        for line in open(filename_train,"r"):
            if line.split(",")[0]!="file":
                OUT_TRAIN.write(line)
        if i != test_id:
            for line in open(filename_test,"r"):
                if line.split(",")[0]!="file":
                    OUT_TRAIN.write(line)

filename_test = PATH+"_iemocap_0"+str(test_id)+f+".test.csv"
for line in open(filename_train,"r"):
    if line.split(",")[0]!="file":
        OUT_TEST.write(line)


