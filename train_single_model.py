import argparse

import torch.optim as optim
import torch.utils.data.sampler as sampler

from models.model_ResNet import MTLDeepLabv3, MTANDeepLabv3
from models.model_SegNet import SegNetSplit, SegNetMTAN
from models.model_EdgeSegNet import EdgeSegNet
from models.model_DDRNet import DualResNet, BasicBlock
from models.model_GuideDepth import GuideDepth
from utils import *
from dataloader import DecnetDataloader
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Single-task Learning: Dense Prediction Tasks')
parser.add_argument('--mode', default='none', type=str)
parser.add_argument('--port', default='none', type=str)

parser.add_argument('--gpu', default=0, type=int, help='gpu ID')
parser.add_argument('--network', default='DDRNet', type=str, help='e.g. SegNet_split, SegNet_mtan, ResNet_split, Resnet_mtan, EdgeSegNet')
parser.add_argument('--dataset', default='sim_warehouse', type=str, help=',sim-warehouse,nyuv2, cityscapes')
parser.add_argument('--task', default='depth', type=str, help='choose task for single task learning')
parser.add_argument('--seed', default=0, type=int, help='gpu ID')

opt = parser.parse_args()

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
random.seed(opt.seed)

# create logging folder to store training weights and losses
if not os.path.exists('logging'):
    os.makedirs('logging')

# define model, optimiser and scheduler
device = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
train_tasks = create_task_flags(opt.task, opt.dataset)
print(train_tasks)
train_tasks = {'depth': 1}#, 'semantic': 23}#, 'normals': 3}


print('Training Task: {} - {} in Single Task Learning Mode with {}'
      .format(opt.dataset.title(), opt.task.title(), opt.network.upper()))

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

total_epoch = 20
 
# define or load optimizer and scheduler
if "ResNet" in opt.network:
    optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)
elif "SegNet" in opt.network:
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
elif "EdgeSegNet" in opt.network:
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.95)
elif "GuidedDepth" in opt.network:
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
elif "DDRNet" in opt.network:
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) # Just winging this one, 
    # should try ty implement the one in original paper


if opt.dataset == 'sim_warehouse':
    batch_size = 8
    #print("\nSTEP. Loading datasets...")
    train_loader = DataLoader(DecnetDataloader('dataset/sim_warehouse/train/datalist_train_warehouse_sim.list', split='train'),batch_size=batch_size,num_workers=0, shuffle=True)
    test_loader = DataLoader(DecnetDataloader('dataset/sim_warehouse/test/datalist_test_warehouse_sim.list', split='eval'),batch_size=1)


# Train and evaluate multi-task network
train_batch = len(train_loader)
test_batch = len(test_loader)

train_metric = TaskMetric(train_tasks, train_tasks, batch_size, total_epoch, opt.dataset)
test_metric = TaskMetric(train_tasks, train_tasks, batch_size, total_epoch, opt.dataset)

for index in range(total_epoch):

    # evaluating train data
    model.train()
    for i,multitaskdata in enumerate((train_loader)):
        print(i,len(train_loader))
        #train_data, train_target = train_dataset.next()
        image = multitaskdata['rgb'].to(device)
        print(image.shape)
        print(multitaskdata['depth'].shape)
        if opt.task == 'depth':
            train_target = multitaskdata['depth'].to(device)
        #train_target = multitaskdata['depth']{task_id: multitaskdata[task_id].to(device) for task_id in train_tasks.keys()}
        #print(train_target.shape)
        #print(train_target)
        train_pred = model(image)
        print(train_pred)
        optimizer.zero_grad()
        print(train_tasks)
        train_loss = [compute_loss(train_pred[i], train_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]

        #train_loss = compute_loss(train_pred[0],train_target,opt.task)
        print(train_loss)
        
        train_loss.backward()
        optimizer.step()

        train_metric.update_metric(train_pred, train_target, train_loss)

    train_str = train_metric.compute_metric()
    train_metric.reset()

    # evaluating test data
    model.eval()
    with torch.no_grad():
        test_dataset = iter(test_loader)
        for k in range(test_batch):
            test_data, test_target = test_dataset.next()
            test_data = test_data.to(device)
            test_target = {task_id: test_target[task_id].to(device) for task_id in train_tasks.keys()}

            test_pred = model(test_data)
            test_loss = [compute_loss(test_pred[i], test_target[task_id], task_id) for i, task_id in enumerate(train_tasks)]

            test_metric.update_metric(test_pred, test_target, test_loss)

    test_str = test_metric.compute_metric()
    test_metric.reset()

    scheduler.step()

    print('Epoch {:04d} | TRAIN:{} || TEST:{} | Best: {} {:.4f}'
          .format(index, train_str, test_str, opt.task.title(), test_metric.get_best_performance(opt.task)))

    task_dict = {'train_loss': train_metric.metric, 'test_loss': test_metric.metric}
    np.save('logging/stl_{}_{}_{}_{}.npy'.format(opt.network, opt.dataset, opt.task, opt.seed), task_dict)





