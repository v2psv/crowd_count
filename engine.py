import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import utility


def train(train_loader, model, criterion, optimizer, epoch):
	train_time = utility.AverageMeter()
	losses = utility.AverageMeter()
	error_mse = utility.AverageMeter()
	error_mae = utility.AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	for i, (idx, input, target, cnt) in enumerate(train_loader):
		input_var = torch.autograd.Variable(input, requires_grad=True)
		target_var = torch.autograd.Variable(target.cuda(async=True)).type(torch.cuda.FloatTensor)

		# compute output
		output = model(input_var)
		loss = criterion(output, target_var)

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# record info
		losses.update(loss.data[0], input.size(0))
		mae, mse = utility.accuracy(output, target_var)
		error_mae.update(mae, input.size(0))
		error_mse.update(mse, input.size(0))
		train_time.update(time.time() - end)
		end = time.time()

	return losses, error_mae, error_mse, train_time


def validate(val_loader, model, criterion, input_img, pred_dmap, truth_dmap):
	test_time = utility.AverageMeter()
	losses = utility.AverageMeter()
	error_mse = utility.AverageMeter()
	error_mae = utility.AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	for i, (idx, input, target, cnt) in enumerate(val_loader):
		input_var = torch.autograd.Variable(input, volatile=True)
		target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True).type(torch.cuda.FloatTensor)
		batch_size = input.size(0)

		# compute output
		output = model(input_var)
		loss = criterion(output, target_var)

		input_img[i*batch_size:(i+1)*batch_size, :, :, :] = input.cpu().numpy().copy()
		truth_dmap[i*batch_size:(i+1)*batch_size, :, :, :] = target.cpu().numpy().copy()
		pred_dmap[i*batch_size:(i+1)*batch_size, :, :, :] = output.data.cpu().numpy().copy()

		# record info
		losses.update(loss.data[0], input.size(0))
		mae, mse = utility.accuracy(output, target_var)
		error_mae.update(mae, input.size(0))
		error_mse.update(mse, input.size(0))
		test_time.update(time.time() - end)
		end = time.time()

	return losses, error_mae, error_mse, test_time
