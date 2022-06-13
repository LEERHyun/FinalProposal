from keras_segmentation.models.unet import vgg_unet


weights = 'corrosionunet.h5'
input_dir = 'path/to/dir'
output_dir = 'path/to/dir'
model = vgg_unet(n_classes=2 ,  input_height=512, input_width=512)
model.load_weights(weights)


model.summary()


from keras_segmentation.predict import predict_multiple

predict_multiple(model=model, inp_dir=input_dir,out_dir=output_dir )





