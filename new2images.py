import numpy as np
#from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import preprocess_input
import pandas as pd
pd.options.display.max_colwidth = 100
import os
preprocessing_function = preprocess_input
import numpy as np
plt.style.use("ggplot")
from tensorflow.keras.applications.vgg19 import VGG19
debug=1


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
H, W = 224, 224
def load_image(path, preprocess=True):
    """Load and preprocess image."""

    x = tf.keras.utils.load_img(path, target_size=(H, W))
    if preprocess:
        x = tf.keras.preprocessing.image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocessing_function(x)
    print(x.shape)
    return x
def find_same_patient_image(data,path,label):
    image_info_file='OCSPhoneImages.csv'
    image_name=path.split('/')[-1].split('.')[0]
    image_path_tmp=data[data['Unique_ID_WL_phone_Camera'] == image_name][
        'Unique_ID_FL_phone_Camera']
    caseId=data[data['Unique_ID_WL_phone_Camera'] == image_name]['Case ID'].to_string(index=False)
    image_path2=image_path_tmp.to_string(index=False)


    tmp=path.split('/')
    if tmp[3]=='cancer':
        print('LABEL: ',1)
        label.append(1)

        return os.path.join('./new_data/phone_fl/cancer', image_path2 + '.jpg'),caseId
    else:
        print('LABEL: ',0)
        label.append(0)
        return os.path.join('./new_data/phone_fl/no_cancer', image_path2 + '.jpg'),caseId
def data_preprocess(imagePaths):
    data=pd.read_csv('OCSPhoneImages.csv')
    images1 = []
    images2=[]
    counter=0
    error_counter=0
    my_train_y = []

    for imagePath in imagePaths:
        print('Length of x: ', len(images1), ' length of y:', len(my_train_y))
        if debug and counter==10:
            break
        try:
            tmp_output=load_image(imagePath)
        except:
            print('read image fail')
            error_counter+=1
            continue
        tmp_output=tmp_output[0]

        imagePath2,caseId=find_same_patient_image(data,imagePath,my_train_y)
        try:
            tmp_output2 = load_image(imagePath2)
        except:
            print('read image fail')
            error_counter+=1
            my_train_y=my_train_y[:-1]
            continue
        tmp_output2 = tmp_output2[0]
        images1.append(tmp_output)
        images2.append(tmp_output2)
        counter+=1

    print('total error file when read training data ',error_counter)
    return images1,images2,my_train_y


data_dir = './new_data/phone_wl'
from imutils import paths
imagePaths = sorted(list(paths.list_images(data_dir)))
import random
random.shuffle(imagePaths)

my_train_images1,my_train_images2,my_train_y = data_preprocess(imagePaths)
print('my_train_y: ',my_train_y)
def train_test_split(my_train,my_train_y):
    my_train_raw = np.array(my_train)
    my_train_y_raw = np.array(my_train_y)

    my_train = my_train_raw[:int((len(my_train_raw) / 10) * 9)]
    my_train_y = my_train_y_raw[:int((len(my_train_y_raw) / 10) * 9)]

    # my_train=keras.layers.concatenate([my_train,my_train], axis=2)
    # print(my_train.shape)
    x_test = my_train_raw[int((len(my_train_raw) / 10) * 9):]
    y_test = my_train_y_raw[int((len(my_train_y_raw) / 10) * 9):]
    return my_train,my_train_y,x_test,y_test

train_images1,train_y,test_images1,test_y=train_test_split(my_train_images1,my_train_y)
train_images2,train_y,test_images2,test_y=train_test_split(my_train_images2,my_train_y)
print('check demensions of train test input_x and input_y: ',train_images1.shape,train_images2.shape,train_y.shape,test_images1.shape,test_images2.shape,test_y.shape)


num_classes = 2
input_shape = (224, 224,3)



learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 200
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [1024, 1024]  # Size of the dense layers of the final classifier



def image_encoder():
    inputs=layers.Input(shape=input_shape)
    myModel1 = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    logits=myModel1(inputs)
    model=keras.Model(inputs=inputs,outputs=logits)
    return model
def create_vit_classifier():
    inputs1 = layers.Input(shape=input_shape)
    inputs2 = layers.Input(shape=input_shape)
    myModel1 = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    myModel2 = image_encoder()
    print('input shape', inputs1.shape)
    for layer in myModel1.layers:
        layer.trainable = True

    x1 = myModel1(inputs1)
    x1=layers.Flatten()(x1)
    x2 = myModel2(inputs2)
    x2 = layers.Flatten()(x2)
    x=tf.keras.layers.Concatenate()([x1, x2])
    x=mlp(x, hidden_units=mlp_head_units, dropout_rate=0.5)
    predictions = layers.Dense(num_classes, activation='softmax')(x)  # New softmax layer
    model=keras.Model(inputs=[inputs1,inputs2],outputs=predictions)
    return model


def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss='SparseCategoricalCrossentropy', metrics=['accuracy']
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=[train_images1,train_images2],
        y=train_y,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)


    return history,model


vit_classifier = create_vit_classifier()
dot_img_file='new2images.png'
vit_classifier.summary()
# tf.keras.utils.plot_model(vit_classifier, to_file=dot_img_file, show_shapes=True)
history,model = run_experiment(vit_classifier)
val_acc = history.history['val_accuracy']
print("val_acc")
print(val_acc)
acc = history.history['accuracy']
print("acc")
print(acc)
val_loss = history.history['val_loss']
print("val_loss")
print(val_loss)
loss = history.history['loss']
print("loss")
print(loss)
yhat = model.predict([test_images1,test_images2])
yhat = np.argmax(yhat, axis=-1)
y_pred=test_y
print('yhat shape: ',yhat.shape)
print('yhat: ',yhat)
print('Confusion Matrix')
# print(confusion_matrix(yhat, y_pred))
# print('Classification Report')
# target_names = ['1', '2']
# print(classification_report(yhat, y_pred, target_names=target_names))
