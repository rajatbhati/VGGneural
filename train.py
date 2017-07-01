from keras.datasets import cifar10
from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model = convnet('vgg_19',weights_path="weights/vgg19_weights.h5", heatmap=False)
model.compile(optimizer=sgd, loss='mse')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()



from PIL import Image
import numpy as np
import scipy
import os
X=[]                           
Y=[]
c=-1
for i in x_train:
	c+=1
    try:
        img=Image.fromarray(img.astype('uint8'),'RGB')
        new_width=224
 		new_height=224
		depth=3
		img = Image.open('somepic.jpg')
		img = img.resize((new_width, new_height), Image.ANTIALIAS)
		img=np.reshape(np.array(list(img.getdata())),(new_width,new_height,depth))
	    X.append(img)
        Y.append(y_train[0])

      

    except:
        continue
    else:
X=np.array(X)
Y=np.array(Y)



