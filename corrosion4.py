from keras_segmentation.models.pspnet import vgg_pspnet
from keras_segmentation.pretrained import pspnet_50_ADE_20K



n_classes = 2
pretrained_model = pspnet_50_ADE_20K()

train_dir = 'path/to/dir'
train_ann = 'path/to/annotationdir'
checkpoint_dir = 'path/to/checkpointdir'

model = vgg_pspnet(n_classes=n_classes)



epochs = 20

model.train(
    train_images =  "D:/Drone/Dataset2/Combine/Train/img/",
    train_annotations = "D:/Drone/Dataset2/Combine/Train/annotation/",
    checkpoints_path = "checkpoints/pspnet" , epochs=20, steps_per_epoch = 400, val_steps_per_epoch=400
)

model.save('corrosionpspnet.h5')