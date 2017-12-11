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
	train_time_record = utility.AverageMeter()
	dmap_loss_record = utility.AverageMeter()
	contex_loss_record = utility.AverageMeter()
	error_mae_record = utility.AverageMeter()
	error_mse_record = utility.AverageMeter()

	# switch to train mode
	model.train()
	end = time.time()

	for i, (idx, img, dmap, contex, perspective) in enumerate(train_loader):
		batch_size = img.size(0)
		input_var = torch.autograd.Variable(img).type(torch.cuda.FloatTensor)
		perspective = torch.autograd.Variable(perspective).type(torch.cuda.FloatTensor)
		target_dmap = torch.autograd.Variable(dmap.cuda()).type(torch.cuda.FloatTensor)
		target_contex = torch.autograd.Variable(contex.cuda()).type(torch.cuda.LongTensor)

		# compute output
		pred_dmap, pred_contex = model(input_var, perspective)
		dmap_loss, contex_loss, dmap_contex_loss = criterion(pred_dmap, pred_contex, target_dmap, target_contex)

		# compute gradient and do SGD step
		optimizer.zero_grad()
		dmap_contex_loss.backward()
		optimizer.step()

		# record info
		pred = np.sum(pred_dmap.data.cpu().numpy(), axis=(1, 2, 3))
		truth = np.sum(target_dmap.data.cpu().numpy(), axis=(1, 2, 3))
		error_mae_record.update(np.mean(np.abs(pred-truth)), batch_size)
		error_mse_record.update(np.mean((pred-truth)**2), batch_size)
		dmap_loss_record.update(dmap_loss.data[0], batch_size)
		contex_loss_record.update(contex_loss.data[0], batch_size)
		train_time_record.update(time.time() - end)
		end = time.time()

	return dmap_loss_record, contex_loss_record, error_mae_record, error_mse_record, train_time_record


def validate(val_loader, model, criterion):
	test_time_record = utility.AverageMeter()
	dmap_loss_record = utility.AverageMeter()
	contex_loss_record = utility.AverageMeter()
	error_mae_record = utility.AverageMeter()
	error_mse_record = utility.AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	result_dmap, result_contex, result_idx = [], [], []

	for i, (idx, img, dmap, contex, perspective) in enumerate(val_loader):
		# batch size must be 1
		batch_size = 1

		input_var = torch.autograd.Variable(img, volatile=True).type(torch.cuda.FloatTensor)
		perspective = torch.autograd.Variable(perspective).type(torch.cuda.FloatTensor)
		target_dmap = torch.autograd.Variable(dmap.cuda(), volatile=True).type(torch.cuda.FloatTensor)
		target_contex = torch.autograd.Variable(contex.cuda()).type(torch.cuda.LongTensor)

		# compute output
		pred_dmap, pred_contex = model(input_var, perspective)
		dmap_loss, contex_loss, dmap_contex_loss = criterion(pred_dmap, pred_contex, target_dmap, target_contex)

        # append predicted results
		result_dmap.append(pred_dmap.data.cpu().numpy()[0,:,:,:])
		result_contex.append(pred_contex.data.cpu().numpy()[0,:,:,:])
		result_idx.append(idx.numpy()[0])

		pred = np.sum(pred_dmap.data.cpu().numpy())
		truth = np.sum(target_dmap.data.cpu().numpy())
		error_mae_record.update(np.abs(pred-truth), batch_size)
		error_mse_record.update((pred-truth)**2, batch_size)
		dmap_loss_record.update(dmap_loss.data[0], batch_size)
		contex_loss_record.update(contex_loss.data[0], batch_size)
		test_time_record.update(time.time() - end)
		end = time.time()

	return dmap_loss_record, contex_loss_record, error_mae_record, error_mse_record, test_time_record, result_dmap, result_contex, result_idx

def train_perspective(train_loader, model, criterion, optimizer):
	train_time_record = utility.AverageMeter()
	loss_record = utility.AverageMeter()

	# switch to train mode
	model.train()
	end = time.time()

	for i, (idx, img, perspective) in enumerate(train_loader):
		batch_size = img.size(0)
		input_var = torch.autograd.Variable(img).type(torch.cuda.FloatTensor)
		target_var = torch.autograd.Variable(perspective.cuda()).type(torch.cuda.FloatTensor)

		# compute output
		pred_perspective = model(input_var)
		loss = criterion(pred_perspective, target_var)

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# record info
		loss_record.update(loss.data[0], batch_size)
		train_time_record.update(time.time() - end)
		end = time.time()

	return loss_record, train_time_record


def validate_perspective(val_loader, model, criterion):
	test_time_record = utility.AverageMeter()
	loss_record = utility.AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	result, result_idx = [], []

	for i, (idx, img, perspective) in enumerate(val_loader):
		# batch size must be 1
		batch_size = 1

		input_var = torch.autograd.Variable(img).type(torch.cuda.FloatTensor)
		target_var = torch.autograd.Variable(perspective.cuda()).type(torch.cuda.FloatTensor)

		# compute output
		pred_perspective = model(input_var)
		loss = criterion(pred_perspective, target_var)

        # append predicted results
		result.append(pred_perspective.data.cpu().numpy()[0,:,:,:])
		result_idx.append(idx.numpy()[0])

		loss_record.update(loss.data[0], batch_size)
		test_time_record.update(time.time() - end)
		end = time.time()

	return loss_record, test_time_record, result, result_idx
