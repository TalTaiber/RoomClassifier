import os
import csv

image_paths = []
room_type = []
room_target = []
style_type = []
style_target = []
budget_type = []
budget_target = []

# re-index budget
budget_legend = ['Low Budget', 'Mid Low Budget',  'Mid High Budget', 'High Budget']

with open(r"C:\Users\PC\UR\room_classifier_houzz_dataset\data.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for ind, row in enumerate(csv_reader):
        if ind==0:
            row0 = row
        elif ind%2==0:
            image_paths.append(("images/" + row[0][row[0].rfind("\\")+1:]).replace(' ', '_'))
            room_type.append(row[1].replace(' ', '_'))
            room_target.append(row[2])
            style_type.append(row[3].replace(' ', '_'))
            style_target.append(row[4])
            budget_type.append(row[5].replace(' ', '_'))
            budget_target.append(budget_legend.index(row[5]))
    print(f'Processed {ind} lines.')

with open(r"C:\Users\PC\UR\room_classifier_houzz_dataset\data_fix.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "room", "room_target", "style", "style_target", "budget", "budget_target"])
    for i in range(len(image_paths)):
        writer.writerow([image_paths[i], room_type[i], room_target[i], style_type[i], style_target[i], budget_type[i], budget_target[i]])

room_legend = []
for i in range(len(set(room_type))):
    room_legend.append(room_type[room_target.index(str(i))])
style_legend = []
for i in range(len(set(style_type))):
    style_legend.append(style_type[style_target.index(str(i))])
budget_legend = []
for i in range(len(set(budget_type))):
    budget_legend.append(budget_type[budget_target.index(i)])

# write legend
with open(r"C:\Users\PC\UR\room_classifier_houzz_dataset\legend.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(room_legend)
    writer.writerow(style_legend)
    writer.writerow(budget_legend)

print(len(room_legend)*len(style_legend)*len(budget_legend))
print("done")
