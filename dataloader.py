import os
import xxlimited
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})
 
class RandomHorizontalFlip(object):
    
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        
    def __call__(self, sample):
        if not _is_pil_image(sample):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(sample)))
        if not _is_pil_image(sample):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(sample)))
        if self.probability < 0.5:
            sample = sample.transpose(Image.FLIP_LEFT_RIGHT)
        return sample
    
    
class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        #image, depth = sample['image'], sample['depth']
        if not _is_pil_image(sample):
            raise TypeError('img should be PIL Image. Got {}'.format(type(sample)))
        if random.random() < self.probability:
            sample = np.asarray(sample)
            sample = Image.fromarray(sample[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return sample
    
def torch_min_max(data):
    minmax = (torch.min(data.float()).item(),torch.mean(data.float()).item(),torch.median(data.float()).item(),torch.max(data.float()).item())
    return minmax

def np_min_max(data):
    minmax = (np.min(data.float()),np.max(data.float()),np.mean(data.float()),np.median(data.float()))
    return minmax


def max_depths():
    max_depths = {
    'nyuv2' : 10.0,
    'kitti': 80.0,
    'nn' : 80.0,
    'warehouse_sim' : 80.0
}
 
def datasets_resolutions():
    resolution_dict =   {
        'nyuv2': {
            'full' : (480, 640),
            'half' : (240, 320),
            'mini' : (224, 224)
        },
        'kitti_res' : {
            'full' : (384, 1280),
            'tu_small' : (128, 416),
            'tu_big' : (228, 912),
            'half' : (192, 640)
        },     
        'nn': {
            'full' : (360, 640),
            'half' : (240, 360),
            'mini' : (120, 160)
        },
        'warehouse_sim': {
            'full' : (720, 1280),
            'half' : (360, 640),
            'mini' : (180, 320)    
        }
    }
    return resolution_dict
    
def crops():#ONLY FULL SCALE NOW
    crops = {
        'kitti' : [128, 381, 45, 1196],
        'nyuv2' : [20, 460, 24, 616],
        'nn' : [4, 356, 16, 624]}
        #'warehouse_sim': [0, 359, 16, 624]}
    


def cspn_nyu_input_crop(rgbd):
    nyu_input_crop_transform = transforms.CenterCrop((228,304))
    resized_rgbd = nyu_input_crop_transform(rgbd)
    #print(resized_rgbd.shape)
    return resized_rgbd    

class DecnetDataloader(Dataset):
    def __init__(self, datalist, split):
        
        #Initialization of class
        self.files = []
        self.data_file = datalist
        self.split = split
        self.root = '.'
        self.dataset_type = 'warehouse_sim'
        self.resolution_dict = datasets_resolutions()
        self.resolution = self.resolution_dict[self.dataset_type]['half']
        self.varying_sparsities = False
        self.augment = True
        
        with open(os.path.join(self.root, self.data_file), 'r') as f:
            data_list = f.read().split('\n')
            for data_row in data_list:
                if len(data_row) == 0:
                    continue
                
                data_columns = data_row.split(' ')
            
                if self.dataset_type == 'nn':
                    self.files.append({
                        "rgb": data_columns[0],
                        "d": data_columns[1],
                        "gt": data_columns[2]
                        })

                elif self.dataset_type == 'nyuv2':
                    self.files.append({
                        "rgb": data_columns[0],
                        "gt": data_columns[1],
                        "random_sampling" : data_columns[2]
                    })
                    
                elif self.dataset_type == 'warehouse_sim':
                    self.files.append({
                        "rgb": data_columns[0],
                        "depth": data_columns[1],
                        "semantic": data_columns[2],
                        "normals" : data_columns[3].replace('normals','normals_npy')+'.npy'
                    })
                    
    def __len__(self):
        #Returns amount of samples
        return len(self.files)
    
    def data_sample(self, file_id, transformed_rgb, transformed_depth, transformed_semantic, transformed_normals):
        #Creating a sample as dict
        sample = {'file': file_id,
                 'rgb': transformed_rgb, 
                 'depth': transformed_depth,
                 'semantic': transformed_semantic,
                 'normals' :transformed_normals}
        return sample

    def decnet_transform(self, file_id, rgb, depth, semantic, normals):        #HAVE NOT IMPLEMENTED RESOLUTION + AUGMENTATIONS
     
        #HAVE NOT IMPLEMENTED RESOLUTION + AUGMENTATIONS        
        
        toPILtransform = transforms.ToPILImage()
        pil_rgb = toPILtransform(rgb)
        pil_depth = toPILtransform(depth)
        pil_semantic = toPILtransform(semantic)
        pil_normals = toPILtransform(normals)
        

        flip_probability = random.random()
        
        if self.augment and self.split == 'train':
                            
            t_rgb = transforms.Compose([
                transforms.Resize(self.resolution),
                RandomHorizontalFlip(flip_probability),
                RandomChannelSwap(0.5),
                transforms.PILToTensor()
            ])

            t_dep = transforms.Compose([
                transforms.Resize(self.resolution),
                RandomHorizontalFlip(flip_probability),
                transforms.PILToTensor()
            ])
               
            transformed_rgb = t_rgb(pil_rgb).to('cuda') / 255.
            transformed_depth = t_dep(pil_depth).type(torch.cuda.FloatTensor)/100.
            transformed_semantic = t_dep(pil_semantic).long()
            transformed_normals = t_dep(pil_normals).type(torch.cuda.FloatTensor)
            
            
            
        else:
                
            t_rgb = transforms.Compose([
                transforms.Resize(self.resolution),
                transforms.PILToTensor()
            ])

            t_dep = transforms.Compose([
                transforms.Resize(self.resolution),
                transforms.PILToTensor()
            ])
    
            transformed_rgb = t_rgb(pil_rgb).to('cuda') / 255.
            transformed_depth = t_dep(pil_depth).type(torch.cuda.FloatTensor)/100.
            transformed_semantic = t_dep(pil_semantic).long()
            transformed_normals = t_dep(pil_normals).type(torch.cuda.FloatTensor)
    
        #one_hot_semantic = torch.nn.functional.one_hot(transformed_semantic.squeeze(),num_classes=23)
        return self.data_sample(file_id, transformed_rgb, transformed_depth, transformed_semantic.squeeze(), transformed_normals)


    def get_sparse_depth(self, dep, num_sample): #Only for NYUv2
        channel, height, width = dep.shape
        assert channel == 1
        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)
        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]
        idx_nnz = idx_nnz[idx_sample[:]]
        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))
        dep_sp = dep * mask.type_as(dep)
        return dep_sp
    
    
    def __getitem__(self, index):
        # Creates one sample of data
        rgb = np.array(Image.open(self.files[index]['rgb']))
        if np.shape(rgb)[2] == 4:
            rgb = rgb[:,:,:3]
        depth = np.array(Image.open(self.files[index]['depth']))
        semantic = np.load(self.files[index]['semantic']).astype(np.float32)
        
        #normals = np.array(Image.open(self.files[index]['normals']))
        normals = np.load(self.files[index]['normals'])#.astype(np.float32)
        
        file_id = self.files[index]['rgb']
        #random_sample_no = self.files[index]['random_sampling'] #Only for NYUv2
        transformed_data_sample = self.decnet_transform(file_id, rgb, depth, semantic, normals)
        return transformed_data_sample
    
    