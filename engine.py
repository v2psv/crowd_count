import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import utility


def train(train_loader, model, criterion, optimizer):
	train_time = utility.AverageMeter()
	dmap_loss = utility.AverageMeter()
	error_mae = utility.AverageMeter()
	error_mse = utility.AverageMeter()

	# switch to train mode
	model.train()
	end = time.time()

<<<<<<< HEAD
	for i, (idx, input, target, cnt) in enumerate(train_loader):
		batch_size = input.size(0)
=======
	for i, (idx, input, target, cnt, pmap) in enumerate(train_loader):
		batch_size = input.size(0)
		# print(input.size(), target.size(), pmap.size())
>>>>>>> 3d67cbbd3c604dd4fe89e3b7b7a892c017205b6b
		input_var = torch.autograd.Variable(input, requires_grad=True).type(torch.cuda.FloatTensor)
		input_pmap = torch.autograd.Variable(pmap, requires_grad=True).type(torch.cuda.FloatTensor)
		target_var = torch.autograd.Variable(target.cuda(async=True)).type(torch.cuda.FloatTensor)

		# compute output
		output = model(input_var, input_pmap)
		loss = criterion(output, target_var)

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# record info
		pred = np.sum(output.data.cpu().numpy(), axis=(1, 2, 3))
		truth = np.sum(target.numpy(), axis=(1, 2, 3))
		error_mae.update(np.mean(np.abs(pred-truth)), batch_size)
		error_mse.update(np.mean((pred-truth)**2), batch_size)
		dmap_loss.update(loss.data[0], batch_size)
		train_time.update(time.time() - end)
		end = time.time()

	return dmap_loss, error_mae, error_mse, train_time


def validate(val_loader, model, criterion):
	test_time = utility.AverageMeter()
	dmap_loss = utility.AverageMeter()
	error_mae = utility.AverageMeter()
	error_mse = utility.AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	pred_dmap, pred_idx = [], []

<<<<<<< HEAD
	for i, (idx, input, target, cnt) in enumerate(val_loader):
		# batch_size = input.size(0)
		# batch size must be 1
		batch_size = 1
=======
	for i, (idx, input, target, cnt, pmap) in enumerate(val_loader):
		batch_size = input.size(0)
>>>>>>> 3d67cbbd3c604dd4fe89e3b7b7a892c017205b6b

		input_var = torch.autograd.Variable(input, volatile=True).type(torch.cuda.FloatTensor)
		input_pmap = torch.autograd.Variable(pmap, requires_grad=True).type(torch.cuda.FloatTensor)
		target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True).type(torch.cuda.FloatTensor)

		# compute output
		output = model(input_var, input_pmap)
		loss = criterion(output, target_var)

		pred_dmap.append(output.data.cpu().numpy()[0,:,:,:])
		pred_idx.append(idx.numpy()[0])

<<<<<<< HEAD
		pred = np.sum(output.data.cpu().numpy())
		truth = np.sum(target.numpy())
=======
		pred = np.sum(output.data.cpu().numpy(), axis=(1, 2, 3))
		truth = np.sum(target.numpy(), axis=(1, 2, 3))
>>>>>>> 3d67cbbd3c604dd4fe89e3b7b7a892c017205b6b

		error_mae.update(np.mean(np.abs(pred-truth)), batch_size)
		error_mse.update(np.mean((pred-truth)**2), batch_size)
		dmap_loss.update(loss.data[0], batch_size)
		test_time.update(time.time() - end)
		end = time.time()

<<<<<<< HEAD
=======

>>>>>>> 3d67cbbd3c604dd4fe89e3b7b7a892c017205b6b
	return dmap_loss, error_mae, error_mse, test_time, pred_dmap, pred_idx
