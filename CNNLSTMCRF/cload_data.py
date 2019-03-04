#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    加载数据
"""
import codecs
import pickle
import numpy as np
from utils import map_item2id


def load_vocs(paths):
    """
    加载vocs
    Args:
        paths: list of str, voc路径
    Returns:
        vocs: list of dict
    """
    vocs = []
    for path in paths:
        with open(path, 'rb') as file_r:
            vocs.append(pickle.load(file_r))
    return vocs


def load_lookup_tables(paths):
    """
    加载lookup tables
    Args:
        paths: list of str, emb路径
    Returns:
        lookup_tables: list of dict
    """
    lookup_tables = []
    for path in paths:
        with open(path, 'rb', encoding='utf-8') as file_r:
            lookup_tables.append(pickle.load(file_r))
    return lookup_tables


def init_data(path, feature_names, vocs, max_len,word_len,word_id, model='train', sep='\t'):
    """
    加载数据
    Args:
        path: str, 数据路径
        feature_names: list of str, 特征名称
        vocs: list of dict
        max_len: int, 句子最大长度
        word_len: int, 单词最大
        model: str, in ('train', 'test')
        sep: str, 特征之间的分割符, default is '\t'
    Returns:
        data_dict: dict
    """
    # print(vocs)
    assert model in ('train', 'test')
    file_r = codecs.open(path, 'r', encoding='utf-8')
    sentences = file_r.read().strip().split('\n\n')
    print(sentences)
    sentence_count = len(sentences)
    print(sentence_count)
    feature_count = len(feature_names)
    data_dict = dict()
    for feature_name in feature_names:
        data_dict[feature_name] = np.zeros(
            (sentence_count, max_len), dtype='int32')
    data_dict['char']=np.zeros((len(sentences),max_len,word_len),dtype='int32')
    if model == 'train':
        data_dict['label'] = np.zeros((len(sentences), max_len), dtype='int32')
    for index, sentence in enumerate(sentences):
        items = sentence.split('\n')
        one_instance_items = []
        char_instance_item=[]
        [one_instance_items.append([]) for _ in range(len(feature_names)+1)]
        # 申请字典空间
        [char_instance_item.append([]) for _ in range(len(items))]
        for item_num,item in enumerate(items):
            feature_tokens = item.split(sep)
            # print(feature_tokens)
            for j in range(feature_count):
                one_instance_items[j].append(feature_tokens[j])
                if j==word_id-1:
                    for num,w in enumerate(feature_tokens[j]):
                        if num==word_len:break
                        char_instance_item[item_num].append(w)
            if model == 'train':
                one_instance_items[-1].append(feature_tokens[-1])
        # print(one_instance_items)
        # print(char_instance_item)
        for i in range(len(feature_names)):
            data_dict[feature_names[i]][index, :] = map_item2id(
                one_instance_items[i], vocs[i], max_len)
        for i in range(len(items)):
            data_dict['char'][index][i]=map_item2id(char_instance_item[i],vocs[-1],word_len)
        if model == 'train':
            data_dict['label'][index, :] = map_item2id(
                one_instance_items[-1], vocs[-2], max_len)
    file_r.close()
    # print(data_dict)
    return data_dict
