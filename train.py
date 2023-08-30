import argparse
import os
import time
from datetime import timedelta, datetime
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (LearningRateScheduler, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.optimizers import Adam
import constants
from generator import segmentationGenerator
from loss import modelLoss
from model import create_Model



DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 2
resnet_type = 50

argparser = argparse.ArgumentParser(description='Training')
argparser.add_argument('-e',
                       '--epochs',
                       default=DEFAULT_EPOCHS,
                       help='number of epochs')
argparser.add_argument('-b',
                       '--batch',
                       default=DEFAULT_BATCH_SIZE,
                       help='batch size')
argparser.add_argument('-r',
                       '--resnet',
                       default=resnet_type,
                       help='resnet type')

# convert string arguments to appropriate type
args = argparser.parse_args()
args.epochs = int(args.epochs)
args.batch = int(args.batch)
args.resnet = constants.EncoderType(int(args.resnet))

print("\nTensorFlow detected the following GPU(s):")
tf.test.gpu_device_name()

print("\n\nSetup start: {}\n".format(time.ctime()))
setup_start = time.time()

# model naming parameter
trainingRunTime = datetime.today().strftime('%Y-%m-%d %H %M %S')

if constants.use_unet:
    Notes = 'seg_UNet_se_d'
else:
    Notes = 'KITTI_Road'

# build loss
lossClass = modelLoss(0.001, 0.85, 512, 512, args.batch)
loss = lossClass.applyLoss
loss2 = lossClass.focal_loss
loss3 = lossClass.dice_plus_focal
# build data generators
train_generator = segmentationGenerator(constants.data_train_image_dir, constants.data_train_gt_dir,
                                        batch_size=args.batch, shuffle=True)
test_generator = segmentationGenerator(constants.data_train_image_dir, constants.data_train_gt_dir,
                                       batch_size=args.batch, shuffle=True, test=True)

# build model
model = create_Model(input_shape=(512, 512, 3), encoder_type=args.resnet)
# model = create_Unet()
model.compile(optimizer=Adam(lr=1e-3), loss=loss2, metrics=['accuracy'])

modelSavePath = 'models/' + Notes + '_' + trainingRunTime + '_batchsize_' + str(args.batch) + '_resnet_' + str(
    args.resnet.value) + '/_weights_epoch{epoch:02d}_val_loss_{val_loss:.4f}_valacc_{val_accuracy:.4f}_train_loss_{loss:.4f}.hdf5'
os.makedirs(
    'D:/segmentation-of-remote-sensing-image-test/acc_and_loss/' + Notes + '_' + trainingRunTime + '_batchsize_' + str(
        args.batch) + '_resnet_' + str(args.resnet.value) + '/')

imsavepath1 = 'D:/segmentation-of-remote-sensing-image-test/acc_and_loss/' + Notes + '_' + trainingRunTime + '_batchsize_' + str(
    args.batch) + '_resnet_' + str(args.resnet.value) + '/Training and validation loss.jpg'
imsavepath2 = 'D:/segmentation-of-remote-sensing-image-test/acc_and_loss/' + Notes + '_' + trainingRunTime + '_batchsize_' + str(
    args.batch) + '_resnet_' + str(args.resnet.value) + '/Training and validation acc.jpg'

# callbacks
if not os.path.exists('models/' + Notes + '_' + trainingRunTime + '_batchsize_' + str(args.batch) + '_resnet_' + str(
        args.resnet.value) + '/'):
    os.makedirs('models/' + Notes + '_' + trainingRunTime + '_batchsize_' + str(args.batch) + '_resnet_' + str(
        args.resnet.value) + '/')
mc = ModelCheckpoint(modelSavePath, monitor='val_loss')
mc1 = ModelCheckpoint(modelSavePath, monitor='loss')
rl = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)  # not used
tb = TensorBoard(log_dir='logs/' + Notes + '_' + trainingRunTime + '_batchsize_' + str(args.batch) + '_resnet_' + str(
    args.resnet.value), histogram_freq=0, write_graph=True, write_images=True)


# Schedule Learning rate Callback
def lr_schedule(epoch):
    if epoch >= 0:
        return 1e-4


lr = LearningRateScheduler(schedule=lr_schedule, verbose=1)

print("Model saved to:")
print(modelSavePath)

print("\n\nTraining start: {}\n".format(time.ctime()))
training_start = time.time()

history = model.fit(train_generator, epochs=args.epochs, validation_data=test_generator, callbacks=[mc, mc1, lr, tb],
                    initial_epoch=0)


# 繪製
def draw(history):
    epochs = range(1, len(history.history['loss']) + 1, 1)
    plt.plot(epochs, history.history['loss'], 'r', label='Training loss')
    plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
    best_loss = min(history.history['val_loss'])
    print(best_loss)

    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(imsavepath1)
    plt.show()

    plt.figure()
    epochs = range(1, len(history.history['accuracy']) + 1, 1)
    plt.plot(epochs, history.history['accuracy'], 'r', label='Training acc')
    plt.plot(epochs, history.history['val_accuracy'], 'b', label='validation acc')
    best_acc = max(history.history['val_accuracy'])
    print(best_acc)
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()

    plt.savefig(imsavepath2)

    plt.show()


print("\n\nTraining end:   {}\n".format(time.ctime()))
print("Model saved to: {}".format(modelSavePath))
training_end = time.time()
draw(history)
setup_time = training_start - setup_start
training_time = training_end - training_start

print("Total setup time: {}".format(str(timedelta(seconds=setup_time))))
print("Total train time: {}".format(str(timedelta(seconds=training_time))))
