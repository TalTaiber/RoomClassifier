import os

root = r"C:\Users\PC\UR\www.houzz.com-photos"

file = []
room = []
style = []
budget = []

for d in os.walk(root):
    if d[2]:
        for f in d[2]:
            file.append(os.path.join(d[0], f))
            splits = d[0].split("\\")
            style.append(splits[-1])
            budget.append(splits[-2])
            room.append(splits[-3])

room_op

print("ok")