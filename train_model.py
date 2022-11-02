import argparse

import torch.nn as nn
import torch.optim as optim
import torch.utils.data.sampler as sampler

from models.model_ResNet import MTLDeepLabv3, MTANDeepLabv3
from models.model_SegNet import SegNetSplit, SegNetMTAN
from models.model_EdgeSegNet import EdgeSegNet
from models.model_DDRNet import DualResNet, BasicBlock
from models.model_GuideDepth import GuideDepth
#from create_dataset import *
from utils import *
from dataloader import DecnetDataloader
from tqdm import tqdm

""" Script for training MTL models """
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Multi-task/Auxiliary Learning: Dense Prediction Tasks')

parser.add_argument('--network', default='DDRNet', type=str, help='e.g. SegNet_split, SegNet_mtan, ResNet_split, Resnet_mtan, EdgeSegNet')
parser.add_argument('--weight', default='equal', type=str, help='weighting methods: equal, dwa, uncert')
parser.add_argument('--task', default='all', type=str, help='primary tasks, use all for MTL setting')
parser.add_argument('--dataset', default='sim_warehouse', type=str, help=',sim-warehouse,nyuv2, cityscapes')
parser.add_argument('--seed', default=0, type=int, help='random seed ID')
parser.add_argument('--load_model', action='store_true', help='pass flag to load checkpoint')
parser.add_argument('--grad_method', default='none', type=str, help='graddrop, pcgrad, cagrad')
parser.add_argument('--gpu', default=0, type=int, help='gpu ID')

opt = parser.parse_args()

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

#train_tasks = create_task_flags('all', opt.dataset, with_noise=False)
#print(train_tasks)
train_tasks = {'depth': 1}#, 'semantic': 23}#, 'normals': 3}
pri_tasks = {'depth': 1}#, 'semantic': 23}#, 'normals': 3}

#pri_tasks = create_task_flags(opt.task, opt.dataset, with_noise=False)
#print(pri_tasks)
train_tasks_str = ''.join(task.title() + ' + ' for task in train_tasks.keys())[:-3]
pri_tasks_str = ''.join(task.title() + ' + ' for task in pri_tasks.keys())[:-3]
print('Dataset: {} | Training Task: {} | Primary Task: {} in Multi-task / Auxiliary Learning Mode with {}'
      .format(opt.dataset.title(), train_tasks_str, pri_tasks_str, opt.network.upper()))
print('Applying Multi-task Methods: Weighting-based: {} + Gradient-based: {}'
      .format(opt.weight.title(), opt.grad_method.upper()))

# define new or load excisting model and optimizer 
if opt.network == 'ResNet_split':
    model = MTLDeepLabv3(train_tasks).to(device)
elif opt.network == 'ResNet_mtan':
    model = MTANDeepLabv3(train_tasks).to(device)
elif opt.network == "SegNet_split":
    model = SegNetSplit(train_tasks).to(device)
elif opt.network == "SegNet_mtan":
    model = SegNetMTAN(train_tasks).to(device)
elif opt.network == "EdgeSegNet":
    model = EdgeSegNet(train_tasks).to(device)
elif opt.network == "GuidedDepth":
    model = GuideDepth(train_tasks).to(device) 
elif opt.network == "DDRNet":
    model = DualResNet(BasicBlock, [2, 2, 2, 2], train_tasks, planes=32, spp_planes=128, head_planes=64).to(device)
else:
    raise ValueError    

if opt.load_model == True:
    checkpoint = torch.load(f"models/model_checkpoint_{model_name}_{dataset_name}.pth")
    model.load_state_dict(checkpoint["model_state_dict"]) 

total_epoch = 200

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

# define or load optimizer and scheduler
if "ResNet" in opt.network:
    optimizer = optim.SGD(params, lr=0.1, weight_decay=1e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)
elif "SegNet" in opt.network:
    optimizer = optim.Adam(params, lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
elif "EdgeSegNet" in opt.network:
    optimizer = optim.Adam(params, lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.95)
elif "GuidedDepth" in opt.network:
    optimizer = optim.Adam(params, lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
elif "DDRNet" in opt.network:
    optimizer = optim.SGD(params, lr=0.01, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) # Just winging this one, 
    # should try ty implement the one in original paper

if opt.load_model == True:
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

if opt.dataset == 'sim_warehouse':
    print("\nSTEP. Loading datasets...")
    batch_size = 16
    train_loader = DataLoader(DecnetDataloader('dataset/sim_warehouse/train/datalist_train_warehouse_sim.list', split='train'),batch_size=batch_size,num_workers=0, shuffle=True)#num_workers=0 otherwise there is an error. Need to see why
    eval_loader = DataLoader(DecnetDataloader('dataset/sim_warehouse/test/datalist_test_warehouse_sim.list', split='eval'),batch_size=1)

'''Not implemented yet
elif opt.dataset == 'nyuv2':
    dataset_path = 'dataset/nyuv2'
    train_set = NYUv2(root=dataset_path, train=True, augmentation=True)
    test_set = NYUv2(root=dataset_path, train=False)
    batch_size = 16
'''
    

# Train and evaluate multi-task network
train_batch = len(train_loader)
test_batch = len(eval_loader)

#print(train_batch, test_batch)


train_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.dataset)
test_metric = TaskMetric(train_tasks, pri_tasks, batch_size, total_epoch, opt.dataset, include_mtl=True)

# load loss and initialize index/epoch
if opt.load_model == True:
    loss = checkpoint["loss"]
    index = checkpoint["epoch"] + 1
else:
    index = 0

while index < total_epoch:
    model.train()
    
    for i,multitaskdata in ((enumerate(tqdm(train_loader)))):
        #print(i,len(train_loader))
        image = multitaskdata['rgb'].to(device)
        train_target = {task_id: multitaskdata[task_id].to(device) for task_id in train_tasks.keys()}

        optimizer.zero_grad()
        train_pred = model(image)

        #print(train_tasks)

        train_loss = [compute_loss(train_pred[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]
        train_loss_tmp = [0] * len(train_tasks)

        if opt.weight in ['equal', 'dwa']:
            train_loss_tmp = [w * train_loss[i] for i, w in enumerate(lambda_weight[index])]#torch.tensor([1, 2, 3]))]#

        if opt.weight == 'uncert':
            train_loss_tmp = [1 / (2 * torch.exp(w)) * train_loss[i] + w / 2 for i, w in enumerate(logsigma)]

        loss = sum(train_loss_tmp)
        loss.backward()
        optimizer.step()

        train_metric.update_metric(train_pred, train_target, train_loss)
    train_str = train_metric.compute_metric()
    train_metric.reset()

    model.eval()
    with torch.no_grad():
        for multitaskdatatest in eval_loader:
            image = multitaskdatatest['rgb'].to(device)
            
            test_target = {task_id: multitaskdatatest[task_id].to(device) for task_id in train_tasks.keys()}

            test_pred = model(image)
            test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in
                         enumerate(train_tasks)]

            test_metric.update_metric(test_pred, test_target, test_loss)

    test_str = test_metric.compute_metric()
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
    if index == total_epoch - 1:
        print("Saving full model")
        path = f"models/model_{model_name}_{dataset_name}.pth"
        device = torch.device("cuda")
        model.to(device)
        torch.save(model.state_dict(), path)
        break

    index += 1

print("Training complete")