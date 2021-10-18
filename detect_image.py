# USAGE
# python detect_image.py --model frcnn-resnet --image images/example_01.jpg --labels coco_classes.pickle

# import the necessary packages
from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import cv2
import pandas as pd
counts_train = []
counts_test=[]
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", type=str, required=True,
	#help="00a1ae8867e0bb89f061679e1cf29e80.jpg")
ap.add_argument("-m", "--model", type=str, default="frcnn-resnet",
	choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"],
	help="name of the object detection model")
ap.add_argument("-l", "--labels", type=str, default="/content/object-detection-pytorch/coco_classes.pickle",
	help="coco_classes.pickle")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the list of categories in the COCO dataset and then generate a
# set of bounding box colors for each class
CLASSES = pickle.loads(open(args["labels"], "rb").read())
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# initialize a dictionary containing model name and it's corresponding 
# torchvision function call
MODELS = {
	"frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
	"frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
	"retinanet": detection.retinanet_resnet50_fpn
}

# load the model and set it to evaluation mode
model = MODELS[args["model"]](pretrained=True, progress=True,
	num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model.eval()


def predict(fname):
	image = cv2.imread(fname)
	orig = image.copy()
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image.transpose((2, 0, 1))
	image = np.expand_dims(image, axis=0)
	image = image / 255.0
	image = torch.FloatTensor(image)
	image = image.to(DEVICE)
	detections = model(image)[0]
	count=0
	for i in range(0, len(detections["boxes"])):
		confidence = detections["scores"][i]
		if confidence > args["confidence"]:
			idx = int(detections["labels"][i])
			#box = detections["boxes"][i].detach().cpu().numpy()
			#(startX, startY, endX, endY) = box.astype("int")
			if (CLASSES[idx] == 'dog') | (CLASSES[idx] == 'cat'):
				count = count + 1
			#label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			#print("[INFO] {}".format(label))
			#cv2.rectangle(orig, (startX, startY), (endX, endY),COLORS[idx], 2)
			#y = startY - 15 if startY - 15 > 15 else startY + 15
			#cv2.putText(orig, label, (startX, y),
			#cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
	print(count)
	return count
train =  pd.read_csv('/content/drive/MyDrive/Colab Notebooks/datasets/tabular/train.csv')
test = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/datasets/tabular/test.csv')
train["file_path"] = train["Id"].apply(lambda identifier: "/content/drive/MyDrive/Colab Notebooks/datasets/Mypet/train/" + identifier + ".jpg")
test["file_path"] = test["Id"].apply(lambda identifier: "/content/drive/MyDrive/Colab Notebooks/datasets/Mypet/test/" + identifier + ".jpg")
for i in train["file_path"]:
    counts_train.append(predict(i))
c= pd.DataFrame(counts_train)
c.to_csv('tr.csv')
for i in test["file_path"]:
    counts_test.append(predict(i))
t= pd.DataFrame(counts_test)
t.to_csv('te.csv')