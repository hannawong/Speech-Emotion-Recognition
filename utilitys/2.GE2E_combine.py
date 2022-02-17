import os
PATH = "../data/embed/"
for root, dirs, files in os.walk(PATH, topdown=False):
   for name in files:
      filename = os.path.join(root, name)
      if filename.endswith("npy"):
          print(filename)
          os.system("mv "+filename+" ../data/GE2E")
    