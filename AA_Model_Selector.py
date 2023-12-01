from __future__ import division
import argparse
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
import six
import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Dropout,
    concatenate
)
from tensorflow.keras.layers import add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

K.set_image_data_format('channels_last')


def _bn_relu(input):
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    activation = Activation("relu")(norm)
    return activation


def _conv_bn_relu(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault(
        "kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault(
        "kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault(
        "kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault(
        "kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(
        round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.00001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    def f(input):
        if is_first_block_of_first_layer:
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception(
                "Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_data_format() == 'channels_last':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        # Modified first 4 layers for ResNet-18, removed initial 7x7 convolution and MaxPooling
        input = Input(shape=input_shape)
        pairwise_conv0 = _conv_bn_relu(
            filters=32, kernel_size=(2, 1), strides=(1, 1))(input)
        pairwise_conv1 = _conv_bn_relu(filters=64, kernel_size=(
            2, 1), strides=(1, 1))(pairwise_conv0)
        pairwise_conv2 = _conv_bn_relu(filters=96, kernel_size=(
            2, 1), strides=(1, 1))(pairwise_conv1)
        pairwise_conv3 = _conv_bn_relu(filters=96, kernel_size=(
            2, 1), strides=(1, 1))(pairwise_conv2)

        block = pairwise_conv3
        filters = 96
        print(repetitions)
        for i, r in enumerate(repetitions):
            print("Build block %d" % i)
            block = _residual_block(
                block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)

        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)

        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

        model = Model(inputs=[input, ], outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 2, 2, 2])


def build_CNN(csv_file):
    print('Build model...')

    model = ResnetBuilder.build_resnet_18((1, 20, 40), 9)

    adam = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(loss={'dense': 'categorical_crossentropy', },
                  optimizer=adam,
                  metrics={'dense': 'accuracy', }
                  )
    model.summary()
    np.random.seed(2)

    input_csv = csv_file
    # Setting the Theme of the data visualizer Seaborn
    sns.set(style="dark", context="notebook", palette="muted")
    train = pd.read_csv("%s" % input_csv)
    Y_train = train['label']
    # Dropping Label Column
    X_train = train.drop(labels=['label'], axis=1)

    graph = sns.countplot(Y_train)
    Y_train.value_counts()
    X_train = X_train.values.reshape(-1, 20, 40, 1)
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=9)
    # Spliting Train and Validate set
    random_seed = 2
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2,
                                                      random_state=random_seed)

    g = plt.imshow(X_train[0][:, :, 0])
    epochs = 30
    batch_size = 40

    learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy',
                                                                   patience=3,
                                                                   verbose=1,
                                                                   factor=0.5,
                                                                   min_lr=0.00001)

    # Fit the model
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[learning_rate_reduction],
                        verbose=2,
                        validation_data=(X_val, Y_val),
                        initial_epoch=0
                        )
    # Save the weights
    model.save_weights('%s_weight.h5' % input_csv)
    # Plot the loss and accuracy curves for training and validation
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r',
               label="validation loss", axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['accuracy'],
               color='b', label="Training accuracy")
    ax[1].plot(history.history['val_accuracy'],
               color='r', label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    plt.savefig('loss_acc.png')


def plot(file_path):
    print(file_path)
    test = pd.read_csv(file_path)
    global f_name
    f_name = file_path
    model_list = {"Q.plant": 0, "Q.bird": 1, "Q.yeast": 2,
                  "Q.mammal": 3, "Q.insect": 4, "Q.pfam": 5, "LG": 6, "WAG": 7, "JTT": 8}
    lb_list = ["Q.plant", "Q.bird", "Q.yeast",
               "Q.mammal", "Q.insect", "Q.pfam", "LG"]
    y_test = test['AlnID']
    y_pred = test['Predict']
    results = [[0] * 9 for i in range(9)]
    count_label = [0] * 9
    sum = 0
    for id in range(len(y_test)):
        i = int(y_test[id])
        j = int(model_list[y_pred[id]])
        results[i][j] += 1
        count_label[i] += 1
        sum += 1
    print(np.sum(results))
    results = results / np.sum(results)
    thresh = 0.0
    for i in range(9):
        for j in range(9):
            if count_label[i] > 0:
                results[i][j] = (sum * results[i][j] * 100) / count_label[i]
                results[i][j] = "%.2f" % results[i][j]
            if results[i][j] > thresh:
                thresh = results[i][j]
    thresh = thresh / 2
    lb_list = ["Q.plant", "Q.bird", "Q.yeast",
               "Q.mammal", "Q.insect", "Q.pfam", "LG", "WAG", "JTT"]
    fig, ax = plt.subplots()
    ax.imshow(results)
    ax.set_xticks(np.arange(9))
    ax.set_yticks(np.arange(9))
    ax.set_xticklabels(labels=lb_list)
    ax.set_yticklabels(lb_list)
    ax.set_facecolor('white')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(9):
        for j in range(9):
            text = ax.text(j, i, results[i][j],
                           horizontalalignment="center",
                           color="black" if results[i][j] > thresh else "white")

    plt.tight_layout()
    plt.title("Confusion matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_%s.png' % f_name)
    plt.show()


def predict_aln(h5model, test_csv):
    print('Load model...')

    model = ResnetBuilder.build_resnet_18((1, 20, 40), 9)

    adam = keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(loss={'dense': 'categorical_crossentropy', },
                  optimizer=adam,
                  metrics={'dense': 'accuracy', }
                  )
    model.summary()
    np.random.seed(2)
    weight_file = h5model
    model.load_weights('%s' % weight_file)
    input_file = test_csv
    test = pd.read_csv("%s" % input_file)
    y_test = test['label']
    # Dropping Label Column
    test = test.drop(labels=['label'], axis=1)
    test = test.values.reshape(-1, 20, 40, 1)
    # predict results
    results = model.predict(test)
    Y_pred = results
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    # select the indix with the maximum probability
    results = np.argmax(results, axis=1)
    model_list = {0: "Q.plant", 1: "Q.bird", 2: "Q.yeast",
                  3: "Q.mammal", 4: "Q.insect", 5: "Q.pfam", 6: "LG", 7: "WAG", 8: "JTT"}
    results = pd.Series(results, name="Label")
    list_results = results.to_numpy()
    print(list_results)
    list_name = ["*"] * len(list_results)
    for id in range(len(list_results)):
        list_name[id] = model_list[list_results[id]]
    print(list_name)
    name_aln = pd.Series(y_test)
    list_out_label = pd.Series(list_name, name="Predict")
    submission = pd.concat(
        [pd.Series(name_aln, name="AlnID"), list_out_label], axis=1)
    submission.to_csv("%s_cnn_aln_results.csv" % input_file, index=False)
    plot("%s_cnn_aln_results.csv" % input_file)


def common_accestor(seq1, seq2, seq3):
    com_seq = ['-'] * len(seq1)
    for i in range(0, len(seq1)):
        com_i = '-'
        if seq1[i] == seq2[i] or seq1[i] == seq3[i]:
            com_i = seq1[i]
        elif seq2[i] == seq3[i]:
            com_i = seq2[i]
        com_seq[i] = com_i

    return com_seq


def process_phyml_file(phyml_input_file):
    print("Process file %s" % phyml_input_file)
    list_aa = ["A", "R", "N", "D", "C", "Q", "E", "G", "H",
               "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    dict_aa = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
               "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19}
    dict_all = {}

    list_aa_combined = [[""] * 20 for i in range(20)]
    list_all = [""] * 801
    id_list = 0
    list_all[id_list] = "label"
    csv_file = open("%s_data.csv" % phyml_input_file, "w", newline='')
    csvwriter = csv.writer(csv_file)
    ret_name = "%s_data.csv" % phyml_input_file
    id_col = 1
    count_prop = 0
    for i in range(20):
        for j in range(20):
            list_aa_combined[i][j] = list_aa[i] + list_aa[j]
            id_list += 1
            list_all[id_list] = list_aa_combined[i][j]
            id_col += 1
            count_prop += 1
    id_list = 1
    for i in range(800):
        list_all[id_list] = "prop%d" % id_list
        id_list += 1
    csvwriter.writerow(list_all)
    label = 0
    if "Q.plant" in phyml_input_file:
        label = 0
    elif "Q.bird" in phyml_input_file:
        label = 1
    elif "Q.yeast" in phyml_input_file:
        label = 2
    elif "Q.mammal" in phyml_input_file:
        label = 3
    elif "Q.insect" in phyml_input_file:
        label = 4
    elif "Q.pfam" in phyml_input_file:
        label = 5
    elif "LG" in phyml_input_file:
        label = 6
    elif "WAG" in phyml_input_file:
        label = 7
    elif "JTT" in phyml_input_file:
        label = 8
    in_file = open(phyml_input_file, "r")
    first_line = in_file.readline()
    taxa_count = int(first_line.split()[0])
    site_count = int(first_line.split()[1])
    data = [""*taxa_count for i in range(taxa_count)]
    count = 0
    for line in in_file:
        if len(line) > 10:
            line = line.split()[1]
            data[count] = line
            count += 1
    out_matrix = [[0]*20 for i in range(20)]
    out_matrix_triplet = [[0]*20 for i in range(20)]
    sum = 0
    loop = max(int(400000/site_count), taxa_count)
    for id in range(loop):
        rand1 = random.randint(0, taxa_count-1)
        rand2 = random.randint(0, taxa_count-1)

        while(rand2 == rand1):
            rand2 = random.randint(0, taxa_count-1)

        rand3 = random.randint(0, taxa_count - 1)
        while (rand3 == rand1) or (rand3 == rand2):
            rand3 = random.randint(0, taxa_count - 1)

        line1 = list(data[rand1])
        line2 = list(data[rand2])
        line3 = list(data[rand3])
        common_ans = common_accestor(line1, line2, line3)
        for id_i in range(site_count):
            if (line1[id_i] in list_aa) and (line2[id_i] in list_aa):
                id1 = dict_aa[line1[id_i]]
                id2 = dict_aa[line2[id_i]]
                out_matrix[id1][id2] += 1
                sum += 1
            if (line1[id_i] in list_aa) and (line3[id_i] in list_aa):
                id1 = dict_aa[line1[id_i]]
                id2 = dict_aa[line3[id_i]]
                out_matrix[id1][id2] += 1
                sum += 1
            if (line2[id_i] in list_aa) and (line3[id_i] in list_aa):
                id1 = dict_aa[line2[id_i]]
                id2 = dict_aa[line3[id_i]]
                out_matrix[id1][id2] += 1
                sum += 1
        for id_i in range(site_count):
            if common_ans[id_i] in list_aa:
                if line1[id_i] in dict_aa:
                    id1 = dict_aa[common_ans[id_i]]
                    id2 = dict_aa[line1[id_i]]
                    out_matrix_triplet[id1][id2] += 1
                    sum += 1
                if line2[id_i] in list_aa:
                    id1 = dict_aa[common_ans[id_i]]
                    id2 = dict_aa[line2[id_i]]
                    out_matrix_triplet[id1][id2] += 1
                    sum += 1
                if line3[id_i] in list_aa:
                    id1 = dict_aa[common_ans[id_i]]
                    id2 = dict_aa[line3[id_i]]
                    out_matrix_triplet[id1][id2] += 1
                    sum += 1
    data = [0]*801
    data_c = 1
    data[0] = lb

    for i in range(20):
        for j in range(20):
            key = list_aa[i] + list_aa[j]
            dict_all[key] = out_matrix[i][j]
            data[data_c] = out_matrix[i][j]
            data_c += 1
    for i in range(20):
        for j in range(20):
            key = list_aa[i] + list_aa[j]
            dict_all[key] = out_matrix_triplet[i][j]
            data[data_c] = out_matrix_triplet[i][j]
            data_c += 1

    csvwriter.writerow(data)
    return ret_name


# call main function
def run(args):
    print(args)
    if args.train != '':
        print("Train CNN model from data")
        build_CNN(args.train)
    elif args.test_dataset != '':
        predict_aln(args.built_model, args.test_dataset)


if __name__ == '__main__':
    print("STARTING CNN MODEL SELECTOR...")
    parser = argparse.ArgumentParser(description='AA ModelSelector')

    parser.add_argument('-predict_aln',
                        type=str)

    parser.add_argument('-test_dataset',
                        type=str)

    parser.add_argument('-built_model',
                        type=str)

    parser.add_argument('-train',
                        type=str)
    args = parser.parse_args(sys.argv[1:])

    run(args)
