## -*- coding:utf-8 -*-
import time, sys
import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from data_folder import DataFolder
from datetime import datetime
import models, engine, utility


def load_checkpoint(checkpoint, model):
    checkpoint = torch.load(checkpoint+'/newest_checkpoint.tar')
    model.load_state_dict(checkpoint['state_dict'])

    print("=> loaded checkpoint '{}')".format(checkpoint))

def init_dataloader(args):
    print('Initializing data...')

    with utility.Timer() as t:
        train_loader = torch.utils.data.DataLoader(DataFolder(args=args, mode='test'),
                        batch_size=1, shuffle=True, num_workers=10, pin_memory=False)
        test_loader = torch.utils.data.DataLoader(DataFolder(args=args, mode='test'),
                        batch_size=1, shuffle=False, num_workers=10, pin_memory=False)

    print('Initializing data loader took %ds' % t.interval)
    return train_loader, test_loader


def main(arg_path):
    # load config
    args = utility.load_params(arg_path)

    # define model and loss function (criterion) and optimizer
    model = models.__dict__[args['model']['arch']](activation=args["model"]["activation"])
    model = torch.nn.DataParallel(model).cuda()
    load_checkpoint('./checkpoint/2017-12-20_12-46-34', model)
    train_loader, test_loader = init_dataloader(args)
    runner =  engine.Engine(None, model, train_loader, test_loader)
    # generate result dataset

    recorder, result = runner.validate_perspect()
    utility.print_info(recorder)

    with h5py.File('pred_perspect.h5', 'a') as hdf:
        for i, x in enumerate(result):
            hdf.create_dataset('test/pred_perspect_2/'+str(i), data=x, compression='gzip', compression_opts=4)

main('config3.json')
