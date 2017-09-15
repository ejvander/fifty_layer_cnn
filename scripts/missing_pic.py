import os

base_offset = 0
num_batch = 0
for i in xrange(0, 2000):
    if not os.path.isfile("output_images/classifier_images/image_train_" + str(i) + ".png"):
        if base_offset == 0:
            base_offset = i // 4
        if base_offset + num_batch == i // 4:
            None
        elif base_offset + num_batch + 1 == i // 4:
            num_batch = i // 4 - base_offset
        else:
            print str(base_offset) + " - size: " + str(num_batch + 1)
            base_offset = 0
            num_batch = 0
            # print str(i) + " - " + str(i//4)
if base_offset > 0:
    print str(base_offset) + " - size: " + str(num_batch + 1)
