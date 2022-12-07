import argparse

import torch.nn as nn
import torch.optim as optim
import torch.utils.data.sampler as sampler

from models.model_ResNet import MTLDeepLabv3, MTANDeepLabv3
from models.model_SegNet import SegNetSplit, SegNetMTAN, SegNetSingle
from models.model_DDRNet import DualResNetMTL, BasicBlock, DualResNetSingle
from models.model_GuideDepth import GuideDepth
from models.model_EdgeSegNet import EdgeSegNet


from utils import *
#from dataloader import DecnetDataloader
from tqdm import tqdm
from autolambda_code import SimWarehouse, NYUv2
import visualizer

import segmentation_models_pytorch as smp 
import wandb



    


""" Script for training MTL models """
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Multi-task/Auxiliary Learning: Dense Prediction Tasks')

#Project settings
parser.add_argument('--project_name', type=str, default='MTLwarehouse', help='Project name')
parser.add_argument('--wandb', action='store_true', help='Use wandb logger')
#Generic settings
parser.add_argument('--seed', default=0, type=int, help='random seed ID')
parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
#Training settings
parser.add_argument('--batch_size', default=4, type=int, help='quite self-xplanatory')
parser.add_argument('--total_epochs', default=100, type=int, help='quite self-xplanatory')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
#Task settings
parser.add_argument('--weight', default='equal', type=str, help='weighting methods: equal, dwa, uncert')
parser.add_argument('--task', default='semantic', type=str,choices='all,semantic,depth,normals', help='tasks for training, use all for MTL setting')
parser.add_argument('--dataset', default='nyuv2', type=str, help='Data from simulated warehouse (sim_warehouse) or NYUv2 indoor data (nyuv2)')
#Network settings
parser.add_argument('--network', default='SegNet', type=str, choices='ResNet,SegNet,DDDRNet',help='Base network')
parser.add_argument('--mtl_architecture', default ='Split',choices=['Split','MTAN'], type=str, help='Split or MTAN mtl architecture')
parser.add_argument('--load_model', action='store_true', help='pass flag to load checkpoint')
parser.add_argument('--grad_method', default='none', type=str, help='graddrop, pcgrad, cagrad')

opt = parser.parse_args()


#Initialize weights and biases logger
if opt.wandb == True:
    print("Started logging in wandb")
    wandb.init(project=str(opt.project_name),entity='wandbdimar',name='{}_{}'.format(str(opt.dataset)[0],opt.network))
    wandb.config.update(opt)


torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

model_name = opt.network
dataset_name = opt.dataset

# create logging folder to store training weights and losses
if not os.path.exists('logging'):
    os.makedirs('logging')

# define model, optimiser and scheduler
device = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")

    
############################### 
#BRANCH_1: MTL vs Single Task
###############################

if opt.task == 'all':
        
    if opt.dataset == 'sim_warehouse':
        train_tasks = {'depth': 1, 'semantic': 23, 'normals': 3}
        pri_tasks = {'depth': 1, 'semantic': 23, 'normals': 3}
    elif opt.dataset == 'nyuv2':
        train_tasks = {'depth': 1, 'semantic': 13, 'normals': 3}
        pri_tasks = {'depth': 1, 'semantic': 13, 'normals': 3}
        
    #train_tasks = create_task_flags('all', opt.dataset, with_noise=False)
    network = opt.network + 'MLT' + opt.mtl_architecture
    print(network)
    
    ############################### 
    #UTILS CREATE NETWORK=
    ###############################
    
    if network == 'ResNetMTL_split':
        model = MTLDeepLabv3(train_tasks).to(device)
    elif network == 'ResNetMTL_mtan':
        model = MTANDeepLabv3(train_tasks).to(device)
    elif network == "SegNetMTL_split":
        model = SegNetSplit(train_tasks).to(device)
    elif network == "SegNetMTL_mtan":
        model = SegNetMTAN(train_tasks).to(device)
    #elif network == "EdgeSegNet":
    #    model = EdgeSegNet(train_tasks).to(device)
    #elif network == "GuidedDepth":
    #    model = GuideDepth(train_tasks).to(device) 
    elif network == "DDRNetMTL":
        model = DualResNetMTL(BasicBlock, [2, 2, 2, 2], train_tasks, planes=32, spp_planes=128, head_planes=64).to(device)
    #elif network == "Segmentation":
    #    model = smp.Unet(
    #        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    #        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #        classes=23,                      # model output channels (number of classes in your dataset)
    #    ).to(device)     
    else:
        raise ValueError 
    
    ###############################
else:
    if opt.task == 'semantic':
        if opt.dataset == 'sim_warehouse':
            train_tasks = {'semantic': 23}
            pri_tasks = {'semantic': 23}
        elif opt.dataset == 'nyuv2':
            train_tasks = {'semantic': 13}
            pri_tasks = {'semantic': 13}
    elif opt.task == 'depth':
        train_tasks = {'depth': 1}
        pri_tasks = {'depth': 1}
    elif opt.task == 'normals':
        train_tasks = {'normals': 3}
        pri_tasks = {'normals': 3}  
        
    network = opt.network + 'Single'
    print(network)
        
    if network == "SegNetSingle":
        model = SegNetSingle(train_tasks).to(device)
    #elif network == "ResNetSingle":
    #    raise Exception("Not implemented")
    elif network == "DDRNetSingle":
        model = DualResNetSingle(BasicBlock, [2, 2, 2, 2], train_tasks, opt.dataset, planes=32, spp_planes=128, head_planes=64).to(device)
            
#pri_tasks = create_task_flags(opt.task, opt.dataset, with_noise=False)
#print(pri_tasks)
train_tasks_str = ''.join(task.title() + ' + ' for task in train_tasks.keys())[:-3]
pri_tasks_str = ''.join(task.title() + ' + ' for task in pri_tasks.keys())[:-3]
print('Dataset: {} | Training Task: {} | Primary Task: {} in Multi-task / Auxiliary Learning Mode with {}'
      .format(opt.dataset.title(), train_tasks_str, pri_tasks_str, opt.network.upper()))
print('Applying Multi-task Methods: Weighting-based: {} + Gradient-based: {}'
      .format(opt.weight.title(), opt.grad_method.upper()))
# define new or load excisting model and optimizer 

total_epoch = opt.total_epochs
saving_epoch = 0

# choose task weighting here
if opt.weight == 'uncert':
    logsigma = torch.tensor([-0.7] * len(train_tasks), requires_grad=True, device=device)
    params = list(model.parameters()) + [logsigma]
    logsigma_ls = np.zeros([total_epoch, len(train_tasks)], dtype=np.float32)

if opt.weight in ['dwa', 'equal']:
    T = 2.0  # temperature used in dwa
    lambda_weight = np.ones([total_epoch, len(train_tasks)])
    #print('lambdaweight',lambda_weight)
    params = model.parameters()
    

#UNIVERSAL OPTIMIZERS AND LR SCHEDULER
optimizer = optim.Adam(params, lr=opt.lr)#, eps=1e-3, amsgrad=True)#, momentum=0.9) 
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,30,40,45], gamma=0.1)


if opt.load_model == True:
    checkpoint = torch.load(f"models/model_{model_name}_{dataset_name}_epoch34.pth")#        path = f"models/model_{model_name}_{dataset_name}_epoch{index}.pth"
    #checkpoint = torch.load(f"models/model_{model_name}_{dataset_name}.pth")
    model.load_state_dict(checkpoint["model_state_dict"]) 

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
print("\nSTEP. Loading datasets...")
#model.load_state_dict(torch.load('models/DDRNet23s_imagenet.pth'), strict=True)

if opt.dataset == 'sim_warehouse':
    '''
    batch_size = opt.batch_size
    train_loader = DataLoader(DecnetDataloader('dataset/sim_warehouse/train/datalist_train_warehouse_sim.list', split='train'),batch_size=batch_size,num_workers=0, shuffle=True)#num_workers=0 otherwise there is an error. Need to see why
    test_loader = DataLoader(DecnetDataloader('dataset/sim_warehouse/test/datalist_test_warehouse_sim.list', split='eval'),batch_size=1)
    '''
    dataset_path = 'dataset/sim_warehouse'
    batch_size = opt.batch_size 
    train_set = SimWarehouse(root=dataset_path, train=True, augmentation=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    test_set = SimWarehouse(root=dataset_path, train=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False
    )    
elif opt.dataset == 'nyuv2':
    dataset_path = 'dataset/nyuv2'
    batch_size = opt.batch_size 
    train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    test_set = NYUv2(root=dataset_path, train=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False
    )    
else:
    raise ValueError
    
#Visualize data for sanity check
train_sample = next(iter(train_loader))  
test_sample = next(iter(test_loader))


data_sample = train_sample

print(f"Data sanity check. RGB.shape: {data_sample['rgb'].shape},\tDepth.shape {data_sample['depth'].shape},\
    \tSemantic.shape {data_sample['semantic'].shape},\tNormals.shape {data_sample['normals'].shape}")
#print(test_sample)#

#sanity_train_target = {task_id: data_sample[task_id].to(device) for task_id in train_tasks.keys()}
#print('sanity_train_target',sanity_train_target.shape)
# Train and evaluate multi-task network
train_batch = len(train_loader)
test_batch = len(test_loader)

#print(train_batch, test_batch)


#train_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.dataset)
#test_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.dataset)#, include_mtl=True)


train_metric = OriginalTaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.dataset)
test_metric = OriginalTaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.dataset)#, include_mtl=True)

# load loss and initialize index/epoch
if opt.load_model == True:
    loss = checkpoint["loss"]
    index = checkpoint["epoch"] + 1
else:
    index = 0


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def min_max_sanity_check(returned_data_dict):
    #print(returned_data_dict["rgb"].to(device))
    print(f'rgb {torch_min_max(returned_data_dict["rgb"])}')
    print(f'depth {torch_min_max(returned_data_dict["depth"])}')
    print(f'semantic {torch_min_max(returned_data_dict["semantic"])}')
    print(f'normals {torch_min_max(returned_data_dict["normals"])}')
 
    

def torch_min_max(data):
    minmax = (torch.min(data.float()).item(),torch.mean(data.float()).item(),torch.median(data.float()).item(),torch.max(data.float()).item())
    return minmax


while index < total_epoch:
    model.train()
    force_cudnn_initialization()
    for i,multitaskdata in ((enumerate(tqdm(train_loader)))):
        
        #print(i,len(train_loader))
        image = multitaskdata['rgb'].to(device)
        #print(multitaskdata)
        train_target = {task_id: multitaskdata[task_id].to(device) for task_id in train_tasks.keys()}

        optimizer.zero_grad()
        #print(image.shape)
        train_pred = model(image)
        
        #print(f'train_pred_shape {train_pred.shape}')
        #print(torch.min(train_pred[0]), torch.max(train_pred[0]))

        if i == 0:
            #print(multitaskdata['file'])
            min_max_sanity_check(multitaskdata)
            #print(f'prediction {torch_min_max(train_pred[0])}')
        
        #print(train_tasks)
        if opt.task == 'all':
            train_loss = [compute_loss(train_pred[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]
            train_loss_tmp = [0] * len(train_tasks)
        else:
            #print('getting in now')
            #print(f'train_pred_shape {train_pred.shape}')
            #print(f'train_target[task_id].shape {train_target["semantic"].shape}')
            

            train_loss = [compute_loss(train_pred, train_target[task_id], task_id) for task_id in train_tasks.keys()]
            train_loss_tmp = [0] * len(train_tasks)


        if opt.weight in ['equal', 'dwa']:
            train_loss_tmp = [w * train_loss[i] for i, w in enumerate(lambda_weight[index])]#torch.tensor([1, 2, 3]))]#

        if opt.weight == 'uncert':
            train_loss_tmp = [1 / (2 * torch.exp(w)) * train_loss[i] + w / 2 for i, w in enumerate(logsigma)]


    
        loss = sum(train_loss_tmp)
        loss.backward()
        optimizer.step()
        
        
        if opt.task == 'all':
            train_metric.update_single_metric(train_pred, train_target, train_loss)
            
        else:
            train_metric.update_single_metric(train_pred, train_target, train_loss[0])
            
    train_str = train_metric.compute_metric()
    train_metric.reset()

    model.eval()
    with torch.no_grad():
        
        i_test = 0
        for multitaskdatatest in test_loader:
            image = multitaskdatatest['rgb'].to(device)
            
            test_target = {task_id: multitaskdatatest[task_id].to(device) for task_id in train_tasks.keys()}

            test_pred = model(image)
            #print(f'test_pred[0]_shape {test_pred[0].shape}')
            if i_test == 0:
                #print(multitaskdatatest['file'])
                #min_max_sanity_check(multitaskdatatest)
                #print(f'prediction {torch_min_max(test_pred[0])}')
                visualizer.save_depth_as_uint8colored(test_pred[0],'results/'+opt.dataset+'/'+multitaskdatatest['file'][0].split('/')[-1]+'.png')
            if i_test == 50:
                visualizer.save_depth_as_uint8colored(test_pred[0],'results/'+opt.dataset+'/'+multitaskdatatest['file'][0].split('/')[-1]+'.png')
            if i_test == 100:
                visualizer.save_depth_as_uint8colored(test_pred[0],'results/'+opt.dataset+'/'+multitaskdatatest['file'][0].split('/')[-1]+'.png')
            if i_test == 150:
                visualizer.save_depth_as_uint8colored(test_pred[0],'results/'+opt.dataset+'/'+multitaskdatatest['file'][0].split('/')[-1]+'.png')
            if i_test == 200:
                visualizer.save_depth_as_uint8colored(test_pred[0],'results/'+opt.dataset+'/'+multitaskdatatest['file'][0].split('/')[-1]+'.png')
            if i_test == 250:
                visualizer.save_depth_as_uint8colored(test_pred[0],'results/'+opt.dataset+'/'+multitaskdatatest['file'][0].split('/')[-1]+'.png')
                  
            #test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in
            #            enumerate(train_tasks)]
            
                    #print(train_tasks)
            if opt.task == 'all':
                test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in
                         enumerate(train_tasks)]
                
                #train_loss = [compute_loss(train_pred[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]
            else:
                #print('getting in now')
                #print(f'train_pred_shape {train_pred.shape}')
                #print(f'train_target[task_id].shape {train_target["semantic"].shape}')
                

                test_loss = [compute_loss(test_pred, test_target[task_id], task_id) for task_id in test_target.keys()]
                
            #print(test_loss)
            if opt.task == 'all':
                test_metric.update_metric(test_pred, test_target, test_loss)
            else:
                test_metric.update_single_metric(test_pred, test_target, test_loss[0])

            i_test +=1

    test_str,depth_loss = test_metric.compute_metric()
    test_metric.reset()

    scheduler.step()
    print('Entering evaluation phase...')
    print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: {} {:.4f}'
          .format(index, train_str, test_str, opt.task.title(), test_metric.get_best_performance(opt.task)))

    if opt.weight in ['dwa', 'equal']:
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': lambda_weight}

        print(get_weight_str(lambda_weight[index], train_tasks))

    if opt.weight == 'uncert':
        logsigma_ls[index] = logsigma.detach().cpu()
        dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric,
                'weight': logsigma_ls}

        print(get_weight_str(1 / (2 * np.exp(logsigma_ls[index])), train_tasks))

    np.save('logging/mtl_dense_{}_{}_{}_{}_{}_{}_.npy'
            .format(opt.network, opt.dataset, opt.task, opt.weight, opt.grad_method, opt.seed), dict)
    np.save('logging/mtl_dense_{}_{}_{}_{}_{}_{}_.txt'
            .format(opt.network, opt.dataset, opt.task, opt.weight, opt.grad_method, opt.seed), dict)

    # Save full model
    current_loss = test_metric.get_best_performance(opt.task)
    if index == 0:
        best_test_str = current_loss
        
    print(current_loss,best_test_str)
    if opt.wandb:
        wandb.log({'current_loss':current_loss}, step = index)


    if opt.task == "all" or opt.task == "depth" or opt.task == "normals":
        if current_loss <= best_test_str:
            save_model = True
    else:
        if current_loss >= best_test_str:
            save_model = True
            
    if save_model == True:
        file = f"models/model_{model_name}_{dataset_name}_epoch{saving_epoch}.pth"
        if os.path.exists(file):
            os.remove(f"models/model_{model_name}_{dataset_name}_epoch{saving_epoch}.pth")
        saving_epoch = index
        best_test_str = current_loss
        print("Saving full model")
        path = f"models/model_{model_name}_{dataset_name}_epoch{index}.pth"
        device = torch.device("cuda")
        model.to(device)
        torch.save({
            'epoch': index,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            }, path)
        #break

    index += 1

print("Training complete")