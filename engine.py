import time
import numpy as np
import pandas as pd
import torch
import torch.optim
import torch.utils.data
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import utility, models
from data_folder_shanghaitech import DataFolder
from loss import MSELoss, FocalLoss2d, L1Loss, OrderLoss, CrossEntropyLoss2d, L2_Grad_Loss, GradientLoss


class Engine(object):
	def __init__(self, args, checkpoint_dir):
		cudnn.benchmark = True

		self.checkpoint_dir = checkpoint_dir
		self.init_dataloader(args)
		self.model, self.optimizer = self.init_model_optimizer(args)

		self.target = args['model']['target']
		assert self.target in ['Density', 'Context', 'Perspect', 'Scene', 'MultiTask', 'ContextPyramid']

		if self.target == 'Density':
			self.criterion = MSELoss().cuda()
			self.recorder_list = ['time', 'density_loss', 'error_mae', 'error_mse']
			self.train_loss = pd.DataFrame(columns=['density_loss', 'error_mae', 'error_mse'])
			self.test_loss  = pd.DataFrame(columns=['density_loss', 'error_mae', 'error_mse'])
			# checkpoint_path = args['model']['pretrained']
			# self.pretrained_model = torch.nn.DataParallel(models.Scene_Embed2(pretrained=True)).cuda()
			# self.pretrained_model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
			# for param in self.pretrained_model.parameters():
			# 	param.requires_grad = False
		elif self.target == 'ContextPyramid':
			self.density_criterion = MSELoss().cuda()
			self.grad_loss = GradientLoss(alpha=1).cuda()
			self.context_criterion = FocalLoss2d(gamma=0, weight=args['model']['context_weight']).cuda()  # shanghaitech_A
			# self.context_criterion = FocalLoss2d(gamma=0, weight=args['model']['context_weight']).cuda() # mall
			# self.context_criterion = FocalLoss2d(gamma=0, weight=args['model']['context_weight']).cuda()    # expo2010
			self.recorder_list = ['time', 'density_loss', 'grad_loss', 'context_loss', 'error_mae', 'error_mse']
			self.train_loss = pd.DataFrame(columns=['density_loss', 'grad_loss', 'context_loss', 'error_mae', 'error_mse'])
			self.test_loss  = pd.DataFrame(columns=['density_loss', 'grad_loss', 'context_loss', 'error_mae', 'error_mse'])

		self.epoch, self.best_record, self.train_loss, self.test_loss = self.load_checkpoint(args['model']['resume'])

	def init_model_optimizer(self, args):
		def weights_init(m):
		    classname = m.__class__.__name__
		    if classname.find('Conv2d') != -1:
		        # torch.nn.init.kaiming_normal(m.weight.data, mode='fan_in')
		        m.weight.data.normal_(0.0, 0.02)
		    elif classname.find('BatchNorm2d') != -1:
		        m.weight.data.normal_(1.0, 0.02)
		        m.bias.data.fill_(0)
		    elif classname.find('Linear') != -1:
		        m.weight.data.normal_(0, 0.02)

		print("Initializing model & optimizer... \r",end="")

		with utility.Timer() as t:
			model = models.__dict__[args['model']['arch']](in_dim=args['data']['img_num_channel'],
														   use_bn=args["model"]["use_bn"],
														   activation=args["model"]["activation"],
														   n_class=args['model']['context_levels'],
														   use_pmap=args['data']['use_pmap'])
			model.apply(weights_init)

			if args['model']['optimizer'] in ['Adagrad', 'Adadelta', 'Adam', 'RMSprop']:
				optimizer = torch.optim.__dict__[args['model']['optimizer']](
									filter(lambda p: p.requires_grad, model.parameters()),
									lr=args['model']['learning_rate'],
									weight_decay=args['model']['weight_decay'])

			model = torch.nn.DataParallel(model).cuda()
		print('Model [%s] & Optimizer [%s] initialized. %ds' % (args['model']['arch'], args['model']['optimizer'], t.interval))
		return model, optimizer

	def load_checkpoint(self, resume):
		if resume:
			checkpoint = torch.load(resume+'/best_checkpoint.tar')
			epoch = checkpoint['epoch']
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])

			with pd.HDFStore(resume + '/loss.h5', 'r') as hdf:
				train_loss = hdf['train_loss']
				test_loss = hdf['test_loss']
			best_record = test_loss[0].min()
			print("=> loaded checkpoint '{}' (epoch {})".format(resume, epoch))
			print("latest train loss %s, test loss %s" % (str(train_loss_t[-1,:]), str(test_loss_t[-1,:])))
		else:
			epoch = 0
			best_record = 9999999
			train_loss = self.train_loss
			test_loss = self.test_loss
		return epoch, best_record, train_loss, test_loss

	def init_dataloader(self, args):
		print("Initializing data... \r",end="")
		with utility.Timer() as t:
			self.train_folder = DataFolder(args=args, mode='train')
			self.test_folder  = DataFolder(args=args, mode='test')
			self.train_loader = torch.utils.data.DataLoader(self.train_folder,
							batch_size=args['data']['train']['batch_size'],
							shuffle=True,
							num_workers=args['data']['train']['num_workers'],
							pin_memory=args['data']['train']['pin_memory'])
			self.test_loader = torch.utils.data.DataLoader(self.test_folder,
							batch_size=args['data']['test']['batch_size'],
							shuffle=False,
							num_workers=args['data']['test']['num_workers'],
							pin_memory=args['data']['test']['pin_memory'])

		print('Initializing data loader took %ds' % t.interval)

	def init_recorder(self, key_list=['time']):
		recorder = {}
		for key in key_list:
			recorder[key] = utility.AverageMeter()
		return recorder

	def update_recorder(self, recorder, pred_density=None, target_density=None, **kwargs):
		if pred_density is not None and target_density is not None:
			batch_size = pred_density.size(0)
			pred = np.sum(pred_density.data.cpu().numpy(), axis=(1, 2, 3))
			truth = np.sum(target_density.data.cpu().numpy(), axis=(1, 2, 3))
			recorder['error_mae'].update(np.mean(np.abs(pred-truth)), batch_size)
			recorder['error_mse'].update(np.mean((pred-truth)**2), batch_size)

		for name, value in kwargs.items():
			batch_size = value.size(0)
			recorder[name].update(value.data[0], batch_size)

		recorder['time'].update(time.time() - self.current_time)
		self.current_time = time.time()

		return recorder

	def update_loss(self, recorder, mode='train'):
		assert mode in ['train', 'test']

		if mode == 'train':
			df_loss = self.train_loss
		elif mode == 'test':
			df_loss = self.test_loss

		n = df_loss.shape[0]
		df_loss.loc[n] = [recorder[x].avg for x in self.train_loss.columns.values]

		with open(self.checkpoint_dir + '/' + mode + '_loss.csv', 'w') as f:
			df_loss.to_csv(f, header=True)

	def save_checkpoint(self, result_dict, recorder):
		status = {'epoch': self.epoch,
				  'optimizer': self.optimizer.state_dict(),
				  'state_dict': self.model.state_dict()
				  }
		utility.save_checkpoint(self.checkpoint_dir, status, mode='newest')
		utility.save_result(self.checkpoint_dir, result_dict=result_dict, mode='newest', num=10)

		if self.target in ['Density', 'MultiTask', 'ContextPyramid']:
			current_record = recorder['error_mae'].avg
		elif self.target == 'Context':
		 	current_record = recorder['context_loss'].avg
		elif self.target == 'Perspect':
			current_record = recorder['perspect_loss'].avg
		elif self.target == 'Scene':
			current_record = recorder['context_loss'].avg + recorder['perspect_loss'].avg

		if current_record < self.best_record:
			self.best_record = current_record
			print('----------------------[Best Record !]----------------------')
			utility.save_checkpoint(self.checkpoint_dir, status, mode='best')
			utility.save_result(self.checkpoint_dir, result_dict=result_dict, mode='best')

	def train_epoch(self):
		self.model.train()
		self.epoch += 1
		recorder = self.init_recorder(self.recorder_list)
		self.current_time = time.time()
		num_iter = len(self.train_loader)

		for i, (idx, img, label) in enumerate(self.train_loader):
			if i % 2 == 0:
				print(f"Training... {i/num_iter*100:.1f} %\r",end="")

			input_var = torch.autograd.Variable(img.cuda(), requires_grad=False).type(torch.cuda.FloatTensor)
			if self.target in ['Density', 'Perspect']:
				label_var = torch.autograd.Variable(label.cuda(), requires_grad=False).type(torch.cuda.FloatTensor)
			elif self.target in ['Context']:
				label_var = torch.autograd.Variable(label.cuda(), requires_grad=False).type(torch.cuda.LongTensor)

			predict, _ = self.model(input_var)
			loss = self.criterion(predict, label_var)

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			if self.target == 'Density':
				self.update_recorder(recorder, density_loss=loss, pred_density=predict, target_density=label_var)
			elif self.target == 'Context':
				self.update_recorder(recorder, context_loss=loss)
			elif self.target == 'Perspect':
				self.update_recorder(recorder, perspect_loss=loss)

		self.update_loss(recorder, 'train')
		utility.print_info(recorder, epoch=self.epoch, preffix='Train ')

	def validate_epoch(self):
		self.model.eval()
		recorder = self.init_recorder(self.recorder_list)
		self.current_time = time.time()
		num_iter = len(self.test_loader)

		result = [None] * len(self.test_loader)
		for i, (idx, img, label) in enumerate(self.test_loader):
			if i % 10 == 0:
				print(f"Validating... {i/num_iter*100:.1f} %\r",end="")

			idx = idx.numpy()[0]

			input_var = torch.autograd.Variable(img.cuda(), requires_grad=False).type(torch.cuda.FloatTensor)
			if self.target in ['Density', 'Perspect'] :
				label_var = torch.autograd.Variable(label.cuda(), requires_grad=False).type(torch.cuda.FloatTensor)
			elif self.target in ['Context']:
				label_var = torch.autograd.Variable(label.cuda(), requires_grad=False).type(torch.cuda.LongTensor)

			predict, _ = self.model(input_var)
			loss = self.criterion(predict, label_var)

			if self.target == 'Density':
				self.update_recorder(recorder, density_loss=loss, pred_density=predict, target_density=label_var)
			elif self.target == 'Context':
				self.update_recorder(recorder, context_loss=loss)
			elif self.target == 'Perspect':
				self.update_recorder(recorder, perspect_loss=loss)

			result[idx] = predict.data.cpu().numpy()[0,:,:,:]

		self.update_loss(recorder, 'test')
		utility.print_info(recorder, preffix='*** Validation *** ')
		self.save_checkpoint(result_dict={self.target:result}, recorder=recorder)

	def test(self):
		self.model.eval()
		num_iter = len(self.test_loader)
		result = [None] * len(self.test_loader)

		for i, (idx, img) in enumerate(self.test_loader):
			if i % 10 == 0:
				print(f"Testing... {i/num_iter*100:.1f} %\r",end="")
			idx = idx.numpy()[0]
			input_var = torch.autograd.Variable(img.cuda(), requires_grad=False).type(torch.cuda.FloatTensor)
			predict = self.model(input_var)
			result[idx] = predict.data.cpu().numpy()[0,:,:,:]

		utility.save_result(self.checkpoint_dir, result_dict={self.target:result}, mode='best')

	def train_pyramid_epoch(self):
		self.model.train()
		self.epoch += 1
		recorder = self.init_recorder(self.recorder_list)
		self.current_time = time.time()
		num_iter = len(self.train_loader)

		x = 0
		for i, (idx, img, density, context) in enumerate(self.train_loader):
			print(f"Training... {i/num_iter*100:.1f} %\r",end="")

			input_var = torch.autograd.Variable(img.cuda(), requires_grad=False).type(torch.cuda.FloatTensor)
			density_var = torch.autograd.Variable(density.cuda(), requires_grad=False).type(torch.cuda.FloatTensor)
			context_var = torch.autograd.Variable(context.cuda(), requires_grad=False).type(torch.cuda.LongTensor)
			# pmap_var = torch.autograd.Variable(pmap.cuda(), requires_grad=False).type(torch.cuda.FloatTensor)

			pred_density, pred_context = self.model(input_var)
			pred_density = pred_density.clamp(min=-10, max=10)

			density_loss = self.density_criterion(pred_density, density_var)
			grad_loss = self.grad_loss(pred_density, density_var)
			context_loss = self.context_criterion(pred_context, context_var)

			loss = density_loss + 0.1*grad_loss + context_loss

			self.optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(self.model.parameters(), 1)
			self.optimizer.step()

			self.update_recorder(recorder, density_loss=density_loss, grad_loss=grad_loss,
								pred_density=pred_density, target_density=density_var,
								context_loss=context_loss)

		self.update_loss(recorder, 'train')
		utility.print_info(recorder, epoch=self.epoch, preffix='Train ')

	def validate_pyramid_epoch(self):
		self.model.eval()
		recorder = self.init_recorder(self.recorder_list)
		self.current_time = time.time()
		num_iter = len(self.test_loader)

		result_density, result_context = [], []

		for i, (idx, img, density, context) in enumerate(self.test_loader):
			print(f"Validating... {i/num_iter*100:.1f} %\r",end="")

			input_var = torch.autograd.Variable(img.cuda(), requires_grad=False).type(torch.cuda.FloatTensor)
			density_var = torch.autograd.Variable(density.cuda(), requires_grad=False).type(torch.cuda.FloatTensor)
			context_var = torch.autograd.Variable(context.cuda(), requires_grad=False).type(torch.cuda.LongTensor)
			# pmap_var = torch.autograd.Variable(pmap.cuda(), requires_grad=False).type(torch.cuda.FloatTensor)

			pred_density, pred_context = self.model(input_var)
			# pred_density = torch.autograd.Variable(roi.cuda(), requires_grad=False).type(torch.cuda.FloatTensor) * pred_density
			density_loss = self.density_criterion(pred_density, density_var)
			grad_loss = self.grad_loss(pred_density, density_var)
			context_loss = self.context_criterion(pred_context, context_var)

			for i in range(idx.size(0)):
				result_density.append(pred_density.data.cpu().numpy()[i,:,:,:])
				result_context.append(pred_context.data.cpu().numpy()[i,:,:,:])

			self.update_recorder(recorder, density_loss=density_loss, grad_loss=grad_loss,
								pred_density=pred_density, target_density=density_var,
								context_loss=context_loss)

		self.update_loss(recorder, 'test')
		utility.print_info(recorder, preffix='*** Validation *** ')
		self.save_checkpoint(result_dict={"density":result_density, "context":result_context}, recorder=recorder)
