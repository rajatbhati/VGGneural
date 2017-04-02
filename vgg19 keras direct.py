from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
#im = preprocess_image_batch(['examples/dog.jpg'],img_size=(256,256), crop_size=(224,224), color_mode="bgr")
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model = convnet('vgg_19',weights_path="weights/vgg19_weights.h5", heatmap=False)
model.compile(optimizer=sgd, loss='mse')
out = model.predict(im)
