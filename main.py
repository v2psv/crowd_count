# -*- coding:utf-8 -*-
import time, sys
import h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from data_folder import ImageFolder
from datetime import datetime
import models, engine, utility


def init_optimizer(args, model):
	if args['model']['optimizer'] == 'SGD':
		return torch.optim.SGD(model.parameters(),
							lr=args['model']['learning_rate'],
                            momentum=args['model']['momentum'],
                            weight_decay=args['model']['weight_decay'])

	elif args['model']['optimizer'] == 'Adagrad':
		return torch.optim.Adagrad(model.parameters(),
								lr=args['model']['learning_rate'],
								weight_decay=args['model']['weight_decay'])

	elif args['model']['optimizer'] == 'Adadelta':
		return torch.optim.Adadelta(model.parameters(),
							lr=args['model']['learning_rate'],
							weight_decay=args['model']['weight_decay'])

	elif args['model']['optimizer'] == 'Adam':
		return torch.optim.Adam(model.parameters(),
						lr=args['model']['learning_rate'],
						weight_decay=args['model']['weight_decay'])

	elif args['model']['optimizer'] == 'RMSprop':
		return torch.optim.RMSprop(model.parameters(),
						lr=args['model']['learning_rate'],
						weight_decay=args['model']['weight_decay'])



def load_checkpoint(resume, model, optimizer, train_loss, test_loss):

	if resume:
		checkpoint = torch.load(resume+'/checkpoint.tar')
		start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])

		with h5py.File(resume + '/result.h5', 'r') as hdf:
			train_loss_t = hdf['train_loss'][:, :]
			test_loss_t = hdf['test_loss'][:, :]
        	train_loss[:train_loss_t.shape[0],:] = train_loss_t[:, :]
        	test_loss[:test_loss_t.shape[0], :] = test_loss_t[:, :]

		print("=> loaded checkpoint '{}' (epoch {})".format(resume, start_epoch))
		print("latest train loss [{}, {}, {}], test loss [{}, {}, {}]".format(\
			train_loss_t[-1,0], train_loss_t[-1,1], train_loss_t[-1,2], \
			test_loss_t[-1,0], test_loss_t[-1,1], test_loss_t[-1,2]))
	else:
		start_epoch = 0

	return start_epoch


def init_dataloader(args):
	print('Initializing data...')

	with utility.Timer() as t:

		train_set = ImageFolder(args=args, data_type='train')
		train_loader = torch.utils.data.DataLoader(train_set,
								batch_size=args['data']['train_batch_size'],
								shuffle=True,
								num_workers=1,
								pin_memory=True)

		test_set = ImageFolder(args=args, data_type='test')
		test_loader = torch.utils.data.DataLoader(test_set,
								batch_size=1,
								shuffle=False,
								num_workers=1,
								pin_memory=True)
	print('Initializing data loader took %ds' % t.interval)
	return train_loader, test_loader


if __name__ == "__main__":
	# load config
	if (len(sys.argv) <= 1):
		raise Exception("configure file must be specified!")
	args = utility.load_params(json_file=sys.argv[1])

	# define model and loss function (criterion) and optimizer
	model = models.__dict__[args['model']['arch']](in_dim=args['data']['img_num_channel'],
												   use_bn=args["model"]["use_bn"],
												   activation=args["model"]["activation"])
	model = torch.nn.DataParallel(model).cuda()
	train_criterion = utility.DmapContexLoss().cuda()
	test_criterion = utility.DmapContexLoss().cuda()
	optimizer = init_optimizer(args, model)
	train_loader, test_loader = init_dataloader(args)

	# (loss, mae, mse, rmse)
	args['model']['epochs'] = int(args['model']['epochs'])
	train_loss = np.zeros((args['model']['epochs'], 4))
	test_loss  = np.zeros((args['model']['epochs'] / args['model']['test_freq'], 4))
	start_epoch = load_checkpoint(args['model']['resume'], model, optimizer, train_loss, test_loss)

	# save args
	best_test_mse = 9999999
	checkpoint_dir = './checkpoint/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	utility.save_args(checkpoint_dir, args)

	# start train
	cudnn.benchmark = True
	for e in range(start_epoch, args['model']['epochs']):
		# train
		ret = engine.train(train_loader, model, train_criterion, optimizer)
		dmap_loss, contex_loss, error_mae, error_mse, train_time = ret
		# lr = utility.adjust_learning_rate(optimizer, e, args['model']['learning_rate'], rate=0.5, period=20)

		utility.print_info(epoch=(e, args['model']['epochs']), train_time=train_time,
						   dmap_loss=dmap_loss, contex_loss=contex_loss,
						   error_mae=error_mae, error_mse=error_mse)
		train_loss[e, :] = [dmap_loss.avg, contex_loss.avg, error_mae.avg, error_mse.avg]

		# validation
		if (e+1) % args['model']['test_freq'] == 0:
			ret = engine.validate(test_loader, model, test_criterion)
			dmap_loss, contex_loss, error_mae, error_mse, test_time, pred_dmap, pred_contex, pred_idx = ret

			utility.print_info(test_time=test_time, dmap_loss=dmap_loss,
							   contex_loss=contex_loss, error_mae=error_mae,
							   error_mse=error_mse)
			test_loss[e/args['model']['test_freq']] = [dmap_loss.avg, contex_loss.avg, error_mae.avg, error_mse.avg]

			status = {'epoch': e, 'arch': args['model']['arch'], 'state_dict': model.state_dict(), 'optimizer' : optimizer.state_dict()}

			utility.save_checkpoint(checkpoint_dir, status, mode='newest')
			utility.save_pred_result(checkpoint_dir, train_loss[:e,:], test_loss[:e/args['model']['test_freq'],:], pred_dmap, pred_contex, pred_idx, sample=20, mode='newest')
			if error_mse.avg < best_test_mse:
				best_test_mse = error_mse.avg
				print('----------------------[Best MSE !]----------------------')
				utility.save_checkpoint(checkpoint_dir, status, mode='best')
				utility.save_pred_result(checkpoint_dir, train_loss[:e,:], test_loss[:e/args['model']['test_freq'],:], pred_dmap, pred_contex, pred_idx, mode='best')
