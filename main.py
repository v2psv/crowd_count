# -*- coding:utf-8 -*-
import time, sys
import h5py
import numpy as np
from datetime import datetime
import engine, utility


if __name__ == "__main__":
	# load config
	if (len(sys.argv) <= 1):
		raise Exception("configure file must be specified!")
	args = utility.load_params(json_file=sys.argv[1])
	print('------------------------------------------------------------')
	print('Model: ' + args['model']['arch'])
	print('Target: ' + str(args['model']['target']))
	print('BatchNorm: ' + str(args['model']['use_bn']))
	print('Optimizer: ' + args['model']['optimizer'])
	print('Activation: ' + args['model']['activation'])
	print('Resume: ' + str(args['model']['resume']))

	print('Batch Size: ' + str(args['data']['train']['batch_size']))
	print('Patch Size: ' + str(args['data']['train']['patch_size']))
	print('------------------------------------------------------------')

	checkpoint_dir = '/mnt/pami/zhouqi/checkpoint/' + str(args['model']['target']) + '/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	utility.save_args(checkpoint_dir, args)
	logger = utility.Tee(checkpoint_dir + '/log.txt', 'a')

	runner =  engine.Engine(args, checkpoint_dir)

	num_epoches = int(args['model']['epochs'])
	test_freq = args['model']['test_freq']

	for e in range(runner.epoch, num_epoches):
		runner.train_pyramid_epoch()
		if (e+1) % test_freq == 0:
			runner.validate_pyramid_epoch()
