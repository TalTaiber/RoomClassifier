import os

root = r"C:\Users\PC\UR\room_classifier_houzz_dataset\images"
os.chdir(root)

for f in os.listdir(root):
    r = f.replace(" ", "_")
    if r != f:
        os.rename(f, r)
