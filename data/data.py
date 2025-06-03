import os
import torch
from torch.utils.data import Dataset
from data.utils import *

class Dataset_Generation(Dataset):
    def __init__(self, camo_data, search_data=None, task=None, image_size=512, mode='train', count=1):
        self.camo_data = camo_data
        self.search_data = search_data
        self.image_size = image_size
        self.mode = mode
        self.count = count
        self.image_pairs = self.data_type(task)
        self.length = len(self.image_pairs)

    def data_type(self, task):
        if(task=='cod'):
            return(self.cod_data())
        if(task=='ref-cod'):
            return(self.refcod_data())
        if(task=='rcod'):
            return(self.rcod_data())

    def cod_data(self):
        config = load_config()['dataset']
        cod_img_dir = os.listdir(config[self.camo_data]['Images'])
        cod_mask_dir = os.listdir(config[self.camo_data]['GT'])
        cod_img_dict={}
        cod_mask_dict={}
        for i in range(len(cod_img_dir)):
            cod_img = os.listdir("{}/{}".format(config[self.camo_data]['Images'], cod_img_dir[i]))
            cod_mask = os.listdir("{}/{}".format(config[self.camo_data]['GT'], cod_mask_dir[i]))
            for m in range(len(cod_img)):
                cod_img[m] = '{}/{}/{}'.format(config[self.camo_data]['Images'], cod_img_dir[i], cod_img[m])
                cod_mask[m] = '{}/{}/{}'.format(config[self.camo_data]['GT'], cod_mask_dir[i], cod_mask[m])
            cod_img_dict[i] = cod_img
            cod_mask_dict[i] = cod_mask
        cod_img_dict = {key: sorted(value) for key, value in cod_img_dict.items()}
        cod_mask_dict = {key: sorted(value) for key, value in cod_mask_dict.items()}
        final_pair = []
        for l in range(len(cod_img_dict)):
            for m in range(len(cod_img_dict[l])):
                final_dict = {'cod_img':cod_img_dict[l][m], 'cod_mask':cod_mask_dict[l][m]}
                final_pair.append(final_dict)
        return final_pair
    
    def refcod_data(self):
        config = load_config()['dataset']
        cod_img_dir = os.listdir(config[self.camo_data]['Images'])
        cod_mask_dir = os.listdir(config[self.camo_data]['GT'])
        si_img_dir = os.listdir(config[self.search_data]['Images'])
        si_mask_dir = os.listdir(config[self.search_data]['GT'])
        si_img_dir_update=[]
        intersection = list(set(cod_img_dir) & set(si_img_dir))
        cod_img_dir = intersection
        cod_mask_dir = intersection
        si_img_dir = intersection
        si_mask_dir = intersection
        cod_img_dict={}
        cod_mask_dict={}
        si_img_dict={}
        si_mask_dict={}
        for i in range(len(cod_img_dir)):
            cod_img = os.listdir("{}/{}".format(config[self.camo_data]['Images'], cod_img_dir[i]))
            cod_mask = os.listdir("{}/{}".format(config[self.camo_data]['GT'], cod_mask_dir[i]))
            si_img = os.listdir("{}/{}".format(config[self.search_data]['Images'], si_img_dir[i]))
            si_mask = os.listdir("{}/{}".format(config[self.search_data]['GT'], si_mask_dir[i]))
            for m in range(len(cod_img)):
                cod_img[m] = '{}/{}/{}'.format(config[self.camo_data]['Images'], cod_img_dir[i], cod_img[m])
                cod_mask[m] = '{}/{}/{}'.format(config[self.camo_data]['GT'], cod_mask_dir[i], cod_mask[m])
            for m in range(len(si_img)):
                si_img[m] = '{}/{}/{}'.format(config[self.search_data]['Images'], cod_img_dir[i], si_img[m])
                si_mask[m] = '{}/{}/{}'.format(config[self.search_data]['GT'], si_mask_dir[i], si_mask[m])
            cod_img_dict[i] = cod_img
            cod_mask_dict[i] = cod_mask
            si_img_dict[i] = si_img
            si_mask_dict[i] = si_mask
        cod_img_dict = {key: sorted(value) for key, value in cod_img_dict.items()}
        cod_mask_dict = {key: sorted(value) for key, value in cod_mask_dict.items()}
        si_img_dict = {key: sorted(value) for key, value in si_img_dict.items()}
        si_mask_dict = {key: sorted(value) for key, value in si_mask_dict.items()}
        final_pair = []
        pos_final_pair=[]
        for i in range(len(cod_img_dict)):
            cod_ith_category = cod_img_dict[i]
            for j in range(len(cod_ith_category)):
                cod_img = cod_ith_category[j]
                cod_mask = cod_mask_dict[i][j]
                temp_count = len(si_img_dict[i])
                if(temp_count>self.count):
                    pairs = list(zip(si_img_dict[i], si_mask_dict[i]))
                    random_pairs = random.sample(pairs, k=self.count)
                    pos_img = [pair[0] for pair in random_pairs]
                    pos_mask = [pair[1] for pair in random_pairs]
                else:
                    pairs = list(zip(si_img_dict[i], si_mask_dict[i]))
                    random_pairs = random.sample(pairs, k=len(si_img_dict[i]))
                    pos_img = [pair[0] for pair in random_pairs]
                    pos_mask = [pair[1] for pair in random_pairs]
                for l in range(len(pos_img)):
                    final_dict = {'cod_img':cod_img, 'si_img':pos_img[l], 'cod_mask':cod_mask, 'si_mask':pos_mask[l], 'si_label':i, 'cod_label':i}
                    pos_final_pair.append(final_dict)
                    final_pair.append(final_dict)
        print(len(final_pair))
        return final_pair

    def rcod_data(self):
        config = load_config()['dataset']
        cod_img_dir = os.listdir(config[self.camo_data]['Images'])
        cod_mask_dir = os.listdir(config[self.camo_data]['GT'])
        si_img_dir = os.listdir(config[self.search_data]['Images'])
        si_mask_dir = os.listdir(config[self.search_data]['GT'])
        
        si_img_dir_update=[]
        for k in cod_img_dir:
            for l in si_img_dir:
                if(k==l):
                    si_img_dir_update.append(l)
        si_img_dir = si_img_dir_update
        si_mask_dir = si_img_dir_update
        cod_img_dict={}
        cod_mask_dict={}
        si_img_dict={}
        si_mask_dict={}
        for i in range(len(cod_img_dir)):
            cod_img = os.listdir("{}/{}".format(config[self.camo_data]['Images'], cod_img_dir[i]))
            cod_mask = os.listdir("{}/{}".format(config[self.camo_data]['GT'], cod_mask_dir[i]))
            si_img = os.listdir("{}/{}".format(config[self.search_data]['Images'], si_img_dir[i]))
            si_mask = os.listdir("{}/{}".format(config[self.search_data]['GT'], si_mask_dir[i]))
            for m in range(len(cod_img)):
                cod_img[m] = '{}/{}/{}'.format(config[self.camo_data]['Images'], cod_img_dir[i], cod_img[m])
                cod_mask[m] = '{}/{}/{}'.format(config[self.camo_data]['GT'], cod_mask_dir[i], cod_mask[m])
            for m in range(len(si_img)):
                si_img[m] = '{}/{}/{}'.format(config[self.search_data]['Images'], cod_img_dir[i], si_img[m])
                si_mask[m] = '{}/{}/{}'.format(config[self.search_data]['GT'], si_mask_dir[i], si_mask[m])
            cod_img_dict[i] = cod_img
            cod_mask_dict[i] = cod_mask
            si_img_dict[i] = si_img
            si_mask_dict[i] = si_mask
        cod_img_dict = {key: sorted(value) for key, value in cod_img_dict.items()}
        cod_mask_dict = {key: sorted(value) for key, value in cod_mask_dict.items()}
        si_img_dict = {key: sorted(value) for key, value in si_img_dict.items()}
        si_mask_dict = {key: sorted(value) for key, value in si_mask_dict.items()}
        si_img_dict_train = {}
        si_img_dict_test = {}
        si_mask_dict_train = {}
        si_mask_dict_test = {}
        for key, value_list in si_img_dict.items():
            split_point = int(len(value_list) * 0.8)
            si_img_dict_train[key] = value_list[:split_point]
            si_img_dict_test[key] = value_list[split_point:]
        for key, value_list in si_mask_dict.items():
            split_point = int(len(value_list) * 0.8)
            si_mask_dict_train[key] = value_list[:split_point]
            si_mask_dict_test[key] = value_list[split_point:]
        if(self.mode=='train'):
            si_img_dict = si_img_dict_train
            si_mask_dict = si_mask_dict_train
        else:
            si_img_dict = si_img_dict_test
            si_mask_dict = si_mask_dict_test
        #import pdb;pdb.set_trace()
        final_pair = []
        pos_final_pair=[]
        neg_final_pair=[]
        count = self.count
        for i in range(len(cod_img_dict)):
            cod_ith_category = cod_img_dict[i]
            for j in range(len(cod_ith_category)):
                randn_cate=[]
                neg_sample_img=[]
                neg_sample_mask=[]
                cod_img = cod_ith_category[j]
                cod_mask = cod_mask_dict[i][j]
                temp_count = len(si_img_dict[i])
                while(len(randn_cate)<self.count):
                    rand_num = random.randint(list(cod_img_dict.keys())[0], list(cod_img_dict.keys())[-1])
                    if rand_num != i and rand_num not in randn_cate:
                        randn_cate.append(rand_num)
                for k in range(len(randn_cate)):
                    msk_lst = random.choice(si_mask_dict[randn_cate[k]]).split('-')
                    img_lst = random.choice(si_img_dict[randn_cate[k]]).split('-')
                    num = img_lst[-1].split('.')[0]
                    neg_sample_mask.append({randn_cate[k]: '{}-{}-{}-{}.png'.format(msk_lst[0], msk_lst[1], msk_lst[2], num)})
                    neg_sample_img.append({randn_cate[k]:  '{}-{}-{}-{}.jpg'.format(img_lst[0], img_lst[1], img_lst[2], num)})
                if(temp_count>count):
                    pairs = list(zip(si_img_dict[i], si_mask_dict[i]))
                    random_pairs = random.sample(pairs, k=count)
                    pos_img = [pair[0] for pair in random_pairs]
                    pos_mask = [pair[1] for pair in random_pairs]
                    pairs = list(zip(neg_sample_img, neg_sample_mask))
                    random_pairs = random.sample(pairs, k=count)
                    neg_img = [pair[0] for pair in random_pairs]
                    neg_mask = [pair[1] for pair in random_pairs]
                else:
                    pairs = list(zip(si_img_dict[i], si_mask_dict[i]))
                    random_pairs = random.sample(pairs, k=len(si_img_dict[i]))
                    pos_img = [pair[0] for pair in random_pairs]
                    pos_mask = [pair[1] for pair in random_pairs]
                    pairs = list(zip(neg_sample_img, neg_sample_mask))
                    random_pairs = random.sample(pairs, k=len(si_img_dict[i]))
                    neg_img = [pair[0] for pair in random_pairs]
                    neg_mask = [pair[1] for pair in random_pairs]
                    
                for l in range(len(pos_img)):
                    final_dict = {'cod_img':cod_img, 'si_img':pos_img[l], 'cod_mask':cod_mask, 'si_mask':pos_mask[l], 'si_label':i, 'cod_label':i}
                    final_dict_2 = {'cod_img':cod_img, 'si_img':list(neg_img[l].values())[0], 'cod_mask':cod_mask, 'si_mask':list(neg_mask[l].values())[0], 'si_label':list(neg_img[l].keys())[0], 'cod_label':i}
                    pos_final_pair.append(final_dict)
                    neg_final_pair.append(final_dict_2)
                    final_pair.append(final_dict)
                    final_pair.append(final_dict_2)
        return final_pair

    def __getitem__(self, index):
        if type(index) == torch.Tensor:
            index = index.item()
        sample = self.image_pairs[index]
        return(sample)
    
    def __len__(self):
        return self.length