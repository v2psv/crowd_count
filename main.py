# -*- coding:utf-8 -*-
import time, h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import utility
from data_folder import ImageFolder
from datetime import datetime
from engine import train, validate
import models

"""
parameters
"""
args = {}

# ucsd
# args['train_batch_size'] = 1
# args['train_size'] = 50
# args['test_batch_size'] = 100
# args['test_size'] = 1200
## args['train_idx'] = [i for i in range(600, 1400)]
# args['train_idx'] = [i for i in range(600, 1400, 16)]
# args['test_idx'] = [i for i in range(600)] + [i for i in range(1400, 2000)]

# args['img_width'] = 240
# args['img_height'] = 160
# args['dmap_width'] = 60
# args['dmap_height'] = 40
# args['img_num_channels'] = 1


# fudan
# args['train_batch_size'] = 1
# args['train_size'] = 50
# args['test_batch_size'] = 100
# args['test_size'] = 1000
# # args['train_idx'] = [i for i in range(0, 100)] + [i for i in range(300, 400)] + \
# # 					  [i for i in range(600, 700)] + [i for i in range(900, 1000)] + \
# # 					  [i for i in range(1200, 1300)]
# args['train_idx'] = [i for i in range(0, 100, 10)] + [i for i in range(300, 400, 10)] + \
# 					[i for i in range(600, 700, 10)] + [i for i in range(900, 1000, 10)] + \
# 					[i for i in range(1200, 1300, 10)]
# args['test_idx'] = [i for i in range(100, 300)] + [i for i in range(400, 600)] + \
# 					[i for i in range(700, 900)] + [i for i in range(1000, 1200)] + \
# 					[i for i in range(1300, 1500)]

# mall
args['train_batch_size'] = 1
args['train_size'] = 50
args['test_batch_size'] = 100
args['test_size'] = 1200
# args['train_idx'] = [i for i in range(0, 800)]
args['train_idx'] = [i for i in range(0, 800, 16)]
args['test_idx'] = [i for i in range(800, 2000)]

args['img_width'] = 640
args['img_height'] = 480
args['dmap_width'] = 160
args['dmap_height'] = 120
args['img_num_channels'] = 1

args['arch'] = 'MCNN'
args['epochs'] = 100000
args['test_freq'] = 10
args['learning_rate'] = 1e-2
args['momentum'] = 0.9
args['weight_decay'] = 1e-5

args['dataset_path'] = '/home/zhouqi/crowd_count/dataset/mall_dataset.h5'
args['dmap_group'] = 'dmap_gaussRange_25_sigma_20'
# args['transform'] = ['mask_0', 'scale_160_240', 'normalize_0_0']
# args['augment'] = ['None', 'Invert', 'HorizonFlip', 'RandomPosCrop_80_120', 'RandomPosCrop_120_160']
args['transform'] = ['mask_0', 'normalize_0_255']
args['augment'] = ['None']
args['resume'] = None


def load_checkpoint(resume, model, optimizer, train_loss, test_loss):

	if resume:
		checkpoint = torch.load(resume+'/checkpoint.tar')
		start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])

		with h5py.File(resume + '/result.h5', 'r') as hdf:
			train_loss_t = hdf['train_loss']
        	test_loss_t = hdf['test_loss']
        	train_loss[:train_loss_t.shape[0],:] = train_loss_t[:, :]
        	test_loss[:test_loss_t.shape[0], :] = test_loss_t[:, :]

		print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
		print("latest train loss [{}, {}, {}], test loss [{}, {}, {}]".format(\
			train_loss_t[-1,0], train_loss_t[-1,1], train_loss_t[-1,2], \
			test_loss_t[-1,0], test_loss_t[-1,1], test_loss_t[-1,2]))
	else:
		start_epoch = 1

	return start_epoch


def init_dataloader(args):
	print('Initializing data...')
	with utility.Timer() as t:
		train_set = ImageFolder(dataset_path=args['dataset_path'],
								idx_list=args['train_idx'], dmap_group=args['dmap_group'],
								transform=args['transform'], augment=args['augment'])
		train_loader = torch.utils.data.DataLoader(train_set, batch_size=args['train_batch_size'], \
								shuffle=True, num_workers=1, pin_memory=True)

		test_set = ImageFolder(dataset_path=args['dataset_path'],
							   idx_list=args['test_idx'], dmap_group=args['dmap_group'],
		                       transform=args['transform'])
		test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['test_batch_size'], \
								shuffle=False, num_workers=1, pin_memory=True)
	print('Initializing data took %ds' % t.interval)
	return train_loader, test_loader


if __name__ == "__main__":

	test_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	checkpoint_dir = './checkpoint/' + test_time

	# define model and loss function (criterion) and optimizer
	model = models.__dict__[args['arch']]()
	model = torch.nn.DataParallel(model).cuda()
	criterion = utility.MSELoss().cuda()
	optimizer = torch.optim.Adagrad(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

	train_loss = np.zeros((args['epochs'], 3))
	test_loss  = np.zeros((args['epochs'] / args['test_freq'], 3))
	start_epoch = load_checkpoint(args['resume'], model, optimizer, train_loss, test_loss)
	train_loader, test_loader = init_dataloader(args)

	cudnn.benchmark = True
	utility.save_args(checkpoint_dir, args)

	for e in range(start_epoch, args['epochs']+1):
		loss, error_mae, error_mse, train_time = train(train_loader, model, criterion, optimizer, e)
		utility.print_info(epoch=(e, args['epochs']), train_time=train_time, loss=loss, error_mae=error_mae, error_mse=error_mse)
		train_loss[e-1, :] = [loss.avg, error_mae.avg, error_mse.avg]

		if e % args['test_freq'] == 0:
			input_img = np.zeros((args['test_size'], args['img_num_channels'], args['img_height'], args['img_width']))
			pred_dmap = np.zeros((args['test_size'], 1, args['dmap_height'], args['dmap_width']))
			truth_dmap = np.zeros((args['test_size'], 1, args['dmap_height'], args['dmap_width']))

			loss, error_mae, error_mse, test_time = validate(test_loader, model, criterion, input_img, pred_dmap, truth_dmap)
			utility.print_info(test_time=test_time, loss=loss, error_mae=error_mae, error_mse=error_mse)
			test_loss[e/args['test_freq']-1] = [loss.avg, error_mae.avg, error_mse.avg]

			utility.save_checkpoint(checkpoint_dir,
					{'epoch': e, 'arch': args['arch'],
					'state_dict': model.state_dict(),
					'optimizer' : optimizer.state_dict()})
			utility.save_pred_result(checkpoint_dir, train_loss[:e,:], test_loss[:e/args['test_freq'],:], \
									 input_img, pred_dmap, truth_dmap, sample=20)