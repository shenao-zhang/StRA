import torch
import torchreid
import os
import sys
import torch.nn as nn
import os.path as osp

from torchreid.utils import Logger, compute_model_complexity
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

log_name = 'train.log'
log_name += 'resnet50'
sys.stdout = Logger(osp.join("./log", log_name))

datamanager = torchreid.data.ImageDataManager(
    root='./reid-data',
    sources='market1501',
 #   sources='dukemtmcreid',
    height=256,
    width=128,
    combineall=False,
    batch_size=32,
    transforms=['random_erase', 'random_crop', 'random_flip']
)
model = torchreid.models.build_model(
    name='stru_resnet50',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
)
num_params, flops = compute_model_complexity(model, (1, 3, 256, 128))
print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))
model = nn.DataParallel(model, device_ids=[0, 1]).cuda()

optimizer = torchreid.optim.build_optimizer(
    model, optim='adam', lr=0.0003
)
scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)
engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager, model, None, optimizer, scheduler=scheduler, label_smooth=True
)

engine.run(
    max_epoch=100,
    fixbase_epoch=8,  # 10 when training dukemtmcreid
    open_layers=['layer4', 'global_avg_pool', 'classifier'],
    save_dir='log_StRA',
    print_freq=100
)
