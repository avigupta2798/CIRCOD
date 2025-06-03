from PIL import Image, ImageOps
import numpy as np
import random
import yaml
import torch
import cv2
from torchvision.transforms import ToTensor, Compose, Resize

def load_config():
    return yaml.load(
        open("config.yml", "r"), Loader=yaml.FullLoader
    )

def random_flip(img, label):
    img = ImageOps.flip(img)
    if(label):
        label = ImageOps.flip(label)
    return img, label

def random_mirror(image, label):
    image = ImageOps.mirror(image)
    if(label):
        label = ImageOps.mirror(label)
    return image, label

def random_rotation(image, label):
    mode = Image.BICUBIC
    random_angle = np.random.randint(-15, 15)
    image = image.rotate(random_angle, mode)
    if(label):
        label = label.rotate(random_angle, mode)
    return image, label

def ImageLoader(path):
    image = Image.open(path).convert('RGB')
    return image

def MaskLoader(img):
    img = Image.open(img).convert('L')
    return img

def custom_collate(data):
    return data

def image_preprocess_temp(image, image_size):
    transform = Compose([Resize((image_size, image_size)),ToTensor()])
    image = transform.transforms[1](image)
    shape = image.shape
    padding_x = max(0, shape[1] - shape[2])
    padding_y = max(0, shape[2] - shape[1])
    image = torch.nn.functional.pad(image, (0, padding_x, 0, padding_y))
    return  transform.transforms[0](image), shape

def autoencoder_mask_preprocess_temp(img, label, image_size, mode='train'):
    transform = Compose([Resize((image_size, image_size)),ToTensor()])
    mask_ten = transform.transforms[1](label)
    shape = mask_ten.shape
    padding_x = max(0, shape[1] - shape[2])
    padding_y = max(0, shape[2] - shape[1])
    mask_padded = torch.nn.functional.pad(mask_ten, (0, padding_x, 0, padding_y))
    mask = transform.transforms[0](mask_padded)
    new = img*mask
    new_2 = (1-mask)
    new_3 = 1-img
    new_4 = new_2*new_3
    new_final = new+new_4
    return new_final, mask, mask_ten

def mask_preprocess_temp(img, label, image_size, mode='train', type=None):
    transform = Compose([Resize((image_size, image_size)),ToTensor()])
    mask_orig = transform.transforms[1](label)
    shape = mask_orig.shape
    padding_x = max(0, shape[1] - shape[2])
    padding_y = max(0, shape[2] - shape[1])
    mask = torch.nn.functional.pad(mask_orig, (0, padding_x, 0, padding_y))
    mask = transform.transforms[0](mask)
    return mask, mask_orig

def after_augmentation(image, label):
    image, label = random_flip(image, label)
    image, label = random_mirror(image, label)
    image, label = random_rotation(image, label)
    return image, label

def preprocessing(img, label, image_size, mode, flag=0, type=None):
    img = ImageLoader(img)
    label = MaskLoader(label)
    img, shape = image_preprocess_temp(img, image_size)
    label, mask_orig = mask_preprocess_temp(img, label, image_size, mode=mode, type=type)
    return img, label, mask_orig

def autoencoder_preprocessing(img, label, image_size, mode, flag=1):
    sal_label, bin_label  = None, None
    img = ImageLoader(img)
    label = MaskLoader(label)
    if(flag==1 and mode=='train'):
        img, label = after_augmentation(img, label)
    img, shape = image_preprocess_temp(img, image_size)
    sal_label, bin_label, bin_label_unsized = autoencoder_mask_preprocess_temp(img, label, image_size, mode=mode)
    return img, sal_label, bin_label, bin_label_unsized

def pair_conversion(sample, image_size, mode='train', type=None):
    if mode=='train' or mode=='test':
        sample_new = {key: [d[key] for d in sample] for key in sample[0].keys()}
        sample=sample_new
    sample['cod_mask_orig']=[]
    sample['si_mask_orig']=[]
    sample['name']=[]
    for z in range(len(sample['cod_img'])):
        sample['name'].append(sample['cod_img'][z])
        cod_img, cod_mask, cod_mask_orig = preprocessing(sample['cod_img'][z], sample['cod_mask'][z], image_size, mode=mode, flag=random.randint(0,1), type=type)
        si_img, si_mask, si_mask_orig = preprocessing(sample['si_img'][z], sample['si_mask'][z], image_size, mode=mode, flag=random.randint(0,1))
        sample['cod_img'][z]  = cod_img
        sample['cod_mask'][z] = cod_mask
        sample['si_img'][z]   = si_img
        sample['si_mask'][z]   = si_mask
        sample['cod_mask_orig'].append(cod_mask_orig)
        sample['si_mask_orig'].append(si_mask_orig)
    return sample

def autoencoder_process(sample, image_size, mode='train'):
    if mode=='train' or mode=='test':
        sample_new = {key: [d[key] for d in sample] for key in sample[0].keys()}
        sample=sample_new
    sample['bin_mask']=[]
    sample['name']=[]
    sample['bin_label_unsized']=[]
    for z in range(len(sample['cod_img'])):
        cod_img, sal_mask, bin_mask, bin_label_unsized = autoencoder_preprocessing(sample['cod_img'][z], sample['cod_mask'][z], image_size, mode=mode, flag=random.randint(0,1))
        sample['name'].append(sample['cod_img'][z])
        sample['cod_img'][z]  = cod_img
        sample['cod_mask'][z] = sal_mask
        sample['bin_mask'].append(bin_mask)
        sample['bin_label_unsized'].append(bin_label_unsized)
    return sample

def convert_flow_to_mapping(flow, output_channel_first=True):
    if not isinstance(flow, np.ndarray):
        if len(flow.shape) == 4:
            if flow.shape[1] != 2:
                flow = flow.permute(0, 3, 1, 2)
            B, C, H, W = flow.size()
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if flow.is_cuda:
                grid = grid.cuda()
            map = flow + grid # here also channel first
            if not output_channel_first:
                map = map.permute(0,2,3,1)
        else:
            if flow.shape[0] != 2:
                # size is HxWx2
                flow = flow.permute(2, 0, 1)

            C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if flow.is_cuda:
                grid = grid.cuda()
            map = flow + grid # here also channel first
            if not output_channel_first:
                map = map.permute(1,2,0).float()
        return map.float()
    else:
        # here numpy arrays
        if len(flow.shape) == 4:
            if flow.shape[3] != 2:
                # size is Bx2xHxW
                flow = flow.permute(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = flow.shape[:3]
            map = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                map[i, :, :, 0] = flow[i, :, :, 0] + X
                map[i, :, :, 1] = flow[i, :, :, 1] + Y
            if output_channel_first:
                map = map.transpose(0,3,1,2)
        else:
            if flow.shape[0] == 2:
                # size is 2xHxW
                flow = flow.permute(1,2,0)
            # HxWx2
            h_scale, w_scale = flow.shape[:2]
            map = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            map[:,:,0] = flow[:,:,0] + X
            map[:,:,1] = flow[:,:,1] + Y
            if output_channel_first:
                map = map.transpose(2,0,1).float()
        return map.astype(np.float32)


def convert_mapping_to_flow(map, output_channel_first=True):
    if not isinstance(map, np.ndarray):
        # torch tensor
        if len(map.shape) == 4:
            if map.shape[1] != 2:
                # size is BxHxWx2
                map = map.permute(0, 3, 1, 2)

            B, C, H, W = map.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if map.is_cuda:
                grid = grid.cuda()
            flow = map - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(0,2,3,1)
        else:
            if map.shape[0] != 2:
                # size is HxWx2
                map = map.permute(2, 0, 1)

            C, H, W = map.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if map.is_cuda:
                grid = grid.cuda()

            flow = map - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(1,2,0).float()
        return flow.float()
    else:
        # here numpy arrays
        if len(map.shape) == 4:
            if map.shape[3] != 2:
                # size is Bx2xHxW
                map = map.permute(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = map.shape[:3]
            flow = np.copy(map)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                flow[i, :, :, 0] = map[i, :, :, 0] - X
                flow[i, :, :, 1] = map[i, :, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(0,3,1,2)
        else:
            if map.shape[0] == 2:
                # size is 2xHxW
                map = map.permute(1,2,0)
            # HxWx2
            h_scale, w_scale = map.shape[:2]
            flow = np.copy(map)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            flow[:,:,0] = map[:,:,0]-X
            flow[:,:,1] = map[:,:,1]-Y
            if output_channel_first:
                flow = flow.transpose(2,0,1).float()
        return flow.astype(np.float32)


def remap_using_flow_fields(image, disp_x, disp_y, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT):
    """
    opencv remap : carefull here mapx and mapy contains the index of the future position for each pixel
    not the displacement !
    map_x contains the index of the future horizontal position of each pixel [i,j] while map_y contains the index of the future y
    position of each pixel [i,j]

    All are numpy arrays
    :param image: image to remap, HxWxC
    :param disp_x: displacement on the horizontal direction to apply to each pixel. must be float32. HxW
    :param disp_y: isplacement in the vertical direction to apply to each pixel. must be float32. HxW
    :return:
    remapped image. HxWxC
    """
    remapping=[]
    h_scale, w_scale=image.shape[1:3]
    X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                       np.linspace(0, h_scale - 1, h_scale))
    map_x = (X+disp_x).astype(np.float32)
    map_y = (Y+disp_y).astype(np.float32)
    for i in range(len(image)):
        remapped_image = cv2.remap(image[i].astype(np.uint8), map_x[i], map_y[i], interpolation=interpolation, borderMode=border_mode)
        torch_image = ToTensor()(remapped_image)
        remapping.append(torch_image)
    return remapping