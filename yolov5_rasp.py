import cv2
import numpy as np
import sys
from openvino.inference_engine import IENetwork, IECore
from yolov5_demo import letterbox, YoloParams, parse_yolo_region, intersection_over_union

import time
import datetime

ie = IECore()
net = ie.read_network(model="yolov5n.xml", weights="yolov5n.bin")
net.batch_size = 1

input_blob = next(iter(net.input_info))
n, c, h, w = net.input_info[input_blob].input_data.shape
exec_net = ie.load_network(network=net, device_name="MYRIAD")

frame = cv2.imread("pet-3157961_1280.jpg")
#cv2.imshow("aaa" , frame)
#cv2.waitKey(0)

in_frame = letterbox(frame, (w, h))
in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
in_frame = in_frame.reshape((n, c, h, w))

#print(in_frame)

output_blob_name = list(net.outputs.keys())[0]
#print(output_blob_name)

print(datetime.datetime.now())

time_sta = time.perf_counter() # Timer start
for i in range(1):
    out = exec_net.infer(inputs={input_blob: in_frame})

print(datetime.datetime.now())

time_end = time.perf_counter() # Timer stop
tim = time_end- time_sta

result = exec_net.requests[0].output_blobs
#print(result)
#print(out)

# Collecting object detection results
objects = list()

output = exec_net.requests[0].output_blobs
print(output)

for layer_name, out_blob in output.items():
    layer_params = YoloParams(side=out_blob.buffer.shape[2])
    
    objects += parse_yolo_region(out_blob.buffer, in_frame.shape[2:],
                                    frame.shape[:-1], layer_params,
                                    0.5)

# Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
for i in range(len(objects)):
    if objects[i]['confidence'] == 0:
        continue
    for j in range(i + 1, len(objects)):
        if intersection_over_union(objects[i], objects[j]) > 0.25:
            objects[j]['confidence'] = 0

# Drawing objects with respect to the --prob_threshold CLI parameter
objects = [obj for obj in objects if obj['confidence'] >= 0.25]

with open('data/coco.yaml', 'r') as f:
            labels_map = [x.strip() for x in f]

if len(objects):
    print("\nDetected boxes for batch {}:".format(1))
    print(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX")

origin_im_size = frame.shape[:-1]
#        print(origin_im_size)
for obj in objects:
    # Validation bbox of detected object
    if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
        continue

    det_label = labels_map[obj['class_id'] + 17]
    print(
        "{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} ".format(det_label, 
        obj['confidence'], obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']))
    cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), thickness=3, color=(0, 0, 255))
    cv2.putText(frame, 
                "#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
            (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 1, 1, 2)
    cv2.imwrite('./output.jpg', frame)
print(tim)
