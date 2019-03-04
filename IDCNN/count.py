import yaml
my_config='./config.yml'
with open(my_config, encoding='utf-8') as file_config:
    config = yaml.load(file_config)
labels=[]

with open(config['data_params']['path_result'],encoding='utf-8') as f1:
    for line in f1.readlines():
        line=line.split('\n')[0].split('\t')
        if len(line)!=1:
            if line[len(line)-2] not in labels:
                labels.append(line[len(line)-2])
            if line[len(line)-1] not in labels:
                labels.append(line[len(line)-1])
f=open(config['data_params']['path_result'],encoding='utf-8')
lines=f.readlines()
labelnum=len(labels)
label=[]
counthave=[]
countright=[]
r_label=[]
for i in range(labelnum):
    countright.append(0)
    counthave.append(0)
    r_label.append(0)
for line in lines:
    if line[0]=='\n':continue
    tmp=line.split("\t")
    tlen=len(tmp)
    tmp[tlen-1]=tmp[tlen-1].split("\n")[0]
    '''
    if tmp[tlen-2] not in label:
        label.append(tmp[tlen-2])
    '''
    if tmp[tlen-1] not in label:
        label.append(tmp[tlen-1])
    counthave[labels.index(tmp[tlen-1])]+=1
    if tmp[tlen-1]==tmp[tlen-2]:
        countright[labels.index(tmp[tlen-1])]+=1
    r_label[labels.index(tmp[tlen-2])]+=1
a_cr=0
a_ch=0
a_r=0
for i in range(len(labels)-2):
    r=countright[i]/float(r_label[i])
    if counthave[i]!=0:
        p=countright[i]/float(counthave[i])
    else:
        p=0
    if r+p!=0:
        f=2*r*p/(r+p)
    else:
        f=0
    if labels[i]!='O':
        a_cr+=countright[i]
        a_ch+=counthave[i]
        a_r+=r_label[i]
    print(labels[i],countright[i],counthave[i],r_label[i])
    print('r',r,'p',p,'f',f)

print('p',a_cr/a_r,'r',a_cr/a_ch,'f',2*a_cr*a_cr/((a_cr/a_ch+a_cr/a_r)*a_ch*a_r))


# import os
# root='../data/train/'
# dic={}
# for file in os.listdir(root):
#     if file.split('.')[1]!='ann':continue
#
#     with open(root+file,encoding='utf-8') as f:
#         for line in f.readlines():
#             line=line.split('\n')[0].split('\t')
#             line[1]=line[1].split(' ')
#             line[2]=line[2].replace(' ','')
#             if line[1][0] not in dic:
#                 dic[line[1][0]]={}
#             if line[2] not in dic[line[1][0]]:
#                 dic[line[1][0]][line[2]]=0
#             dic[line[1][0]][line[2]]+=1
#             # print(line)
# # print(dic)
# for i in dic:
#     # print(i)
#     for j in dic[i]:
#         print(i,
#               j,dic[i][j])