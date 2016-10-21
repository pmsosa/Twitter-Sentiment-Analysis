import random
import sys
import os

file_name = sys.argv[1]
count = int(sys.argv[2])

subscript = 1

while os.path.isfile('./good' + str(count) + '_' + str(subscript)):
    subscript += 1

t_file = list(open(file_name, 'r'))
good_file = open("good" + str(count) + '_' + str(subscript), 'a')
bad_file = open("bad" + str(count) + '_' +  str(subscript), 'a')

print("Opened file")

good_count = 0
bad_count = 0

while True:
    line = random.choice(t_file)
    line_split = line.split(',', 2)
    label = int(line_split[1])
    if label and good_count < count:
        good_file.write(line)
        good_count += 1
    elif not label and bad_count < count:
        bad_file.write(line)
        bad_count += 1
    elif bad_count >= count and good_count >= count:
        break