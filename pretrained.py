from keras.datasets import cifar10
from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model = convnet('vgg_19',weights_path="weights/vgg19_weights.h5", heatmap=False)
model.compile(optimizer=sgd, loss='mse')
import PIL
from PIL import Image
new_width=200
new_height=200
depth=3
img = Image.open(image)
img = img.resize((new_width, new_height), Image.ANTIALIAS)
img=np.reshape(np.array(list(img.getdata())),(new_width,new_height,depth))
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)
v=0
import re
for i in P:
	if re.search('car|vehi|train|bus|bike',i[0],re.M|re.I) and p[1] > .3:
		v=1
return v
 