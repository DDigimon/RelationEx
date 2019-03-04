with open('./data/0_train1.txt','w',encoding='utf-8') as fin:
    with open('./data/0_train.txt',encoding='utf-8') as f:
        for line in f.readlines():
            # print(line)
            if line.split('\t')[0] == '':
                fin.write('\n')
            else:
                fin.write(line)