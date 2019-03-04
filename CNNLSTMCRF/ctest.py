#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
    标记文件
"""
import codecs
import yaml
import pickle
import tensorflow as tf
from cload_data import load_vocs, init_data
from cmodel import SequenceLabelingModel


def main():
    # 加载配置文件
    with open('./cconfig.yml') as file_config:
        config = yaml.load(file_config)

    feature_names = config['model_params']['feature_names']

    # 初始化embedding shape, dropouts, 预训练的embedding也在这里初始化
    feature_weight_shape_dict, feature_weight_dropout_dict, \
    feature_init_weight_dict = dict(), dict(), dict()
    char_weight_shape_dict, char_weight_dropout_dict, \
    char_init_weight_dict = dict(), dict(), dict()
    for feature_name in feature_names:
        feature_weight_shape_dict[feature_name] = \
            config['model_params']['embed_params'][feature_name]['shape']
        feature_weight_dropout_dict[feature_name] = \
            config['model_params']['embed_params'][feature_name]['dropout_rate']
        path_pre_train = config['model_params']['embed_params'][feature_name]['path']
        if path_pre_train:
            with open(path_pre_train, 'rb') as file_r:
                feature_init_weight_dict[feature_name] = pickle.load(file_r)
    char_weight_shape_dict['char'] = config['model_params']['embed_params']['char']['shape']
    char_weight_dropout_dict['char'] = config['model_params']['embed_params']['char']['dropout_rate']
    pre_train_char = config['model_params']['embed_params']['char']['path']
    with open(pre_train_char, 'rb') as file_r:
        feature_init_weight_dict['char'] = pickle.load(file_r)

    # 加载数据

    # 加载vocs
    path_vocs = []
    for feature_name in feature_names:
        path_vocs.append(config['data_params']['voc_params'][feature_name]['path'])
    path_vocs.append(config['data_params']['voc_params']['label']['path'])
    path_vocs.append(config['data_params']['voc_params']['char']['path'])
    vocs = load_vocs(path_vocs)

    # 加载数据
    sep_str = config['data_params']['sep']
    assert sep_str in ['table', 'space']
    sep = '\t' if sep_str == 'table' else ' '
    data_dict = init_data(
        path=config['data_params']['path_test'], feature_names=feature_names, sep=sep,
        vocs=vocs, max_len=config['model_params']['sequence_length'],
        word_len=config['model_params']['word_length'],
        word_id=config['model_params']['embed_params']['char']['word_id'],
        model='test')

    # 加载模型
    model = SequenceLabelingModel(
        sequence_length=config['model_params']['sequence_length'],
        word_length=config['model_params']['word_length'],
        nb_classes=config['model_params']['nb_classes'],
        nb_hidden=config['model_params']['bilstm_params']['num_units'],
        feature_weight_shape_dict=feature_weight_shape_dict,
        feature_init_weight_dict=feature_init_weight_dict,
        feature_weight_dropout_dict=feature_weight_dropout_dict,
        char_init_weight_dict=feature_init_weight_dict,
        char_weight_dropout_dict=char_weight_dropout_dict,
        char_weight_shape_dict=char_weight_shape_dict,
        dropout_rate=config['model_params']['dropout_rate'],
        nb_epoch=config['model_params']['nb_epoch'], feature_names=feature_names,
        batch_size=config['model_params']['batch_size'],
        train_max_patience=config['model_params']['max_patience'],
        use_crf=config['model_params']['use_crf'],
        l2_rate=config['model_params']['l2_rate'],
        rnn_unit=config['model_params']['rnn_unit'],
        learning_rate=config['model_params']['learning_rate'],
        clip=config['model_params']['clip'],
        path_model=config['model_params']['path_model'])
    saver = tf.train.Saver()
    saver.restore(model.sess, config['model_params']['path_model'])

    # 标记
    viterbi_sequences = model.predict(data_dict)
    print(viterbi_sequences)

    # 写入文件
    label_voc = dict()
    for key in vocs[-2]:
        label_voc[vocs[-2][key]] = key
    with codecs.open(config['data_params']['path_test'], 'r', encoding='utf-8') as file_r:
        sentences = file_r.read().strip().split('\n\n')
    # print(sentences)
    # print(label_voc)
    file_result = codecs.open(
        config['data_params']['path_result'], 'w', encoding='utf-8')
    for i, sentence in enumerate(sentences):
        for j, item in enumerate(sentence.split('\n')):
            # print(j,item)
            if j < len(viterbi_sequences[i]):
                # print(label_voc[viterbi_sequences[i][j]])
                file_result.write('%s\t%s\n' % (item, label_voc[viterbi_sequences[i][j]]))
            else:
                file_result.write('%s\tO\n' % item)
        file_result.write('\n')

    file_result.close()


if __name__ == '__main__':
    main()
