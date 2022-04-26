import os
import cv2
import yaml
import shutil
import argparse
from tqdm import tqdm
from itertools import chain

import torch 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from data.dataset import CycleGANDataset
from utils import fix_seeds, history_buffer

class LR_Lambda:
    def __init__(self, total_epoch, start_reduce_lr_epoch) -> None:
        self.total_epoch = total_epoch
        self.start_reduce_lr_epoch = start_reduce_lr_epoch

    def __call__(self, epoch):
        if epoch < self.start_reduce_lr_epoch:
            return 1.0
        else:
            return 1.0 - (epoch - self.start_reduce_lr_epoch) / (self.total_epoch - self.start_reduce_lr_epoch)

class CycleGAN:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.cfg_train = self.cfg['train']
        self.cfg_data = cfg['data']
        self.cfg_model = cfg['model']
        self.hitory_buffer_fakeA = history_buffer(50)
        self.hitory_buffer_fakeB = history_buffer(50)

        self.CreateModels(self.cfg_model.get('num_residual_blocks', 9),
                          self.cfg_model['num_hourglass'], 
                          self.cfg_model['use_variant'],
                          self.cfg_train['device'])

        self.CreateOptimizers(cfg_train['lr'])

        self.SetScheduler(self.cfg_train['total_epochs'], 
                          self.cfg_train['start_reduce_lr_epoch'])

        self.CreateDatasets(self.cfg_data['domainA_path'], 
                            self.cfg_data['domainB_path'],
                            self.cfg_data['image_size'],
                            self.cfg_data['crop_size'],
                            self.cfg_data['max_data_per_epoch'],
                            self.cfg_data['name_filter_domainA'],
                            self.cfg_data['name_filter_domainB'])

        self.CreateDataloaders(self.cfg_train['batch_size'])

        self.CreateLossFunctions()

        self.CreateValidationDatasets(self.cfg['val']['data']['domainA_path'], 
                                      self.cfg['val']['data']['domainB_path'],
                                      self.cfg['val']['data']['image_size'],
                                      self.cfg['val']['data']['crop_size'],
                                      self.cfg['val']['num_visualization_img'],
                                      self.cfg['val']['data']['name_filter_domainA'],
                                      self.cfg['val']['data']['name_filter_domainB'])

        self.CreateValidationDataloaders(batch_size=1)

    def CreateModels(self, num_residual_blocks, num_hourglass, use_variant, device='cuda'):
        self.netG_A2B = Generator(num_residual_blocks, num_hourglass, use_variant).to(device)
        self.netG_B2A = Generator(num_residual_blocks, num_hourglass, use_variant).to(device)
        self.netD_A = Discriminator().to(device)
        self.netD_B = Discriminator().to(device)
    
    def CreateOptimizers(self, lr):
        self.optimG = torch.optim.Adam(chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()), lr=lr, betas=(0.5, 0.999))
        self.optimD = torch.optim.Adam(chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=lr, betas=(0.5, 0.999))

    def SetScheduler(self, total_epochs, start_reduce_lr_epoch):
        lr_lambda = LR_Lambda(total_epochs, start_reduce_lr_epoch)
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimG, lr_lambda=lr_lambda)
        self.lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimD, lr_lambda=lr_lambda)
        
    def CreateDatasets(self, domainA_path, domainB_path, image_size, crop_size, max_data_per_epoch, name_filter_domainA = '*.png', name_filter_domainB = '*.png'):
        self.dataset = CycleGANDataset(domainA_path, name_filter_domainA, domainB_path, name_filter_domainB, image_size, crop_size, max_data_per_epoch, mode='train')

    def CreateDataloaders(self, batch_size=1):
        self.dataloader = DataLoader(self.dataset, batch_size, shuffle=True)

    def CreateValidationDatasets(self, domainA_path, domainB_path, image_size, crop_size, max_data_per_epoch, name_filter_domainA = '*.png', name_filter_domainB = '*.png'):
        self.dataset_val = CycleGANDataset(domainA_path, name_filter_domainA, domainB_path, name_filter_domainB, image_size, crop_size, max_data_per_epoch, mode='val')

    def CreateValidationDataloaders(self, batch_size=1):
        self.dataloader_val = DataLoader(self.dataset_val, batch_size, shuffle=False)

    def CreateLossFunctions(self):
        self.cycle_loss = torch.nn.L1Loss()
        self.identity_loss = torch.nn.L1Loss()
        self.adversarial_loss = torch.nn.MSELoss()

    def update_G(self):
        self.optimG.zero_grad()

        # Update A2B phase
        self.fakeB = self.netG_A2B(self.realA)
        cycleA = self.netG_B2A(self.fakeB)

        cycle_loss_A = self.cycle_loss(cycleA, self.realA) * self.cfg['train']['lambda']['cycle']
        score_fakeB_from_netD_B = self.netD_B(self.fakeB)
        adversarial_loss_B = self.adversarial_loss(score_fakeB_from_netD_B, torch.ones_like(score_fakeB_from_netD_B))
        cycle_and_adversarial_loss_GA2B = cycle_loss_A + adversarial_loss_B
        cycle_and_adversarial_loss_GA2B.backward()

        # Update B2A phase
        self.fakeA = self.netG_B2A(self.realB)
        cycleB = self.netG_A2B(self.fakeA)

        cycle_loss_B = self.cycle_loss(cycleB, self.realB) * self.cfg['train']['lambda']['cycle']
        score_fakeA_from_netD_A = self.netD_A(self.fakeA)
        adversarial_loss_A = self.adversarial_loss(score_fakeA_from_netD_A, torch.ones_like(score_fakeA_from_netD_A))
        cycle_and_adversarial_loss_GB2A = cycle_loss_B + adversarial_loss_A
        cycle_and_adversarial_loss_GB2A.backward()

        # Update Identity loss
        if self.cfg['train']['lambda']['identity'] > 0:
            realA2A = self.netG_B2A(self.realA)
            identity_loss_A = self.identity_loss(realA2A, self.realA) * self.cfg['train']['lambda']['identity']
            identity_loss_A.backward()

            realB2B = self.netG_A2B(self.realB)
            identity_loss_B = self.identity_loss(realB2B, self.realB) * self.cfg['train']['lambda']['identity']
            identity_loss_B.backward()
        
        self.optimG.step()

    def get_D_predictions(self):
        self.pred_real_A = self.netD_A(self.realA)
        self.pred_real_B = self.netD_B(self.realB)
        self.fakeA = self.hitory_buffer_fakeA(self.fakeA)
        self.fakeB = self.hitory_buffer_fakeB(self.fakeB)
        # The computational graph is already been released,
        # Detach the fake data to prevent doing the backpropagation on the released computational graph.
        self.pred_fake_A = self.netD_A(self.fakeA.detach())
        self.pred_fake_B = self.netD_B(self.fakeB.detach())

    def update_D(self):
        ''' 1. Compute losses and do back-propagation on two discriminators '''
        loss_real_A = self.adversarial_loss(self.pred_real_A, torch.ones_like(self.pred_real_A))
        loss_real_B = self.adversarial_loss(self.pred_real_B, torch.ones_like(self.pred_real_B))
        loss_fake_A = self.adversarial_loss(self.pred_fake_A, torch.zeros_like(self.pred_fake_A))
        loss_fake_B = self.adversarial_loss(self.pred_fake_B, torch.zeros_like(self.pred_fake_B))
        total_D_losses = (loss_real_A + loss_fake_A) * 0.5 + (loss_real_B + loss_fake_B) * 0.5

        ''' 2. Update two discriminators '''
        self.optimD.zero_grad()
        total_D_losses.backward()
        self.optimD.step()

    def update_scheduler(self):
        self.lr_scheduler_D.step()
        self.lr_scheduler_G.step()

    def save_model(self, epoch):
        check_point = {
            'model_state_dict': {
                'GA2B': self.netG_A2B.state_dict(),
                'GB2A': self.netG_B2A.state_dict(),
                'DA': self.netD_A.state_dict(),
                'DB': self.netD_B.state_dict()
            },
            'optimizer_state_dict': {
                'optimG': self.optimG.state_dict(),
                'optimD': self.optimD.state_dict()
            },
            'scheduler_state_dict': {
                'lr_scheduler_G': self.lr_scheduler_G.state_dict(),
                'lr_scheduler_D': self.lr_scheduler_D.state_dict()
            },
            'epoch': epoch
        }
        if not os.path.isdir('{}/{}/model'.format(self.cfg['output_folder'], self.cfg_train['model_description'])):
            os.makedirs('{}/{}/model'.format(self.cfg['output_folder'], self.cfg_train['model_description']))
        torch.save(check_point, "{}/{}/model/epoch_{}.pth".format(self.cfg['output_folder'], self.cfg_train['model_description'], epoch))

    def load_model(self):
        check_point = torch.load(self.cfg['train']['path_pretrained_model'], map_location=self.cfg['train']['device'])
        # Load models
        print('Loading models...')
        self.netG_A2B.load_state_dict(check_point['model_state_dict']['GA2B'])
        self.netG_B2A.load_state_dict(check_point['model_state_dict']['GB2A'])
        self.netD_A.load_state_dict(check_point['model_state_dict']['DA'])
        self.netD_B.load_state_dict(check_point['model_state_dict']['DB'])
        # Load optimizers
        print('Loading optimizers...')
        self.optimG.load_state_dict(check_point['optimizer_state_dict']['optimG'])
        self.optimD.load_state_dict(check_point['optimizer_state_dict']['optimD'])
        # Load learning rate schedulers
        print('Loading learning rate schedulers...')
        self.lr_scheduler_G.load_state_dict(check_point['scheduler_state_dict']['lr_scheduler_G'])
        self.lr_scheduler_D.load_state_dict(check_point['scheduler_state_dict']['lr_scheduler_D'])

        return check_point['epoch']

    def visualize(self, epoch):
        def get_visualization(tensor):
            tensor = torch.squeeze(tensor)
            tensor = tensor.permute(1, 2, 0)
            tensor = torch.clamp((tensor * 0.5 + 0.5), 0.0, 1.0) * 255
            tensor = tensor.to(dtype=torch.uint8, device='cpu').numpy()
            return tensor

        device = self.cfg['val']['device']
        iters_per_epoch = len(self.dataset_val)
        pbar = tqdm(enumerate(self.dataloader_val), total=iters_per_epoch, desc='Visualizing')
        for i, data in pbar:
            self.optimG.zero_grad()
            self.realA = data['A'].to(device)
            self.realB = data['B'].to(device)
            
            with torch.no_grad():
                pred_A2B = self.netG_A2B(self.realA)
                pred_B2A = self.netG_B2A(self.realB)
            img_pred_A2B = get_visualization(pred_A2B)
            img_pred_B2A = get_visualization(pred_B2A)
            self.realA = get_visualization(self.realA)
            self.realB = get_visualization(self.realB)

            if not os.path.isdir('{}/{}/visualization/A2B/{}'.format(self.cfg['output_folder'], self.cfg_train['model_description'], i)):
                os.makedirs('{}/{}/visualization/A2B/{}'.format(self.cfg['output_folder'], self.cfg_train['model_description'], i))
            if not os.path.isdir('{}/{}/visualization/B2A/{}'.format(self.cfg['output_folder'], self.cfg_train['model_description'], i)):
                os.makedirs('{}/{}/visualization/B2A/{}'.format(self.cfg['output_folder'], self.cfg_train['model_description'], i))

            cv2.imwrite('{}/{}/visualization/A2B/{}/epoch_{}.png'.format(self.cfg['output_folder'], self.cfg_train['model_description'], i, epoch) , img_pred_A2B)
            cv2.imwrite('{}/{}/visualization/B2A/{}/epoch_{}.png'.format(self.cfg['output_folder'], self.cfg_train['model_description'], i, epoch) , img_pred_B2A)
            if epoch == 0:
                cv2.imwrite('{}/{}/visualization/A2B/{}/input.png'.format(self.cfg['output_folder'], self.cfg_train['model_description'], i) , self.realA)
                cv2.imwrite('{}/{}/visualization/B2A/{}/input.png'.format(self.cfg['output_folder'], self.cfg_train['model_description'], i) , self.realB)
            

    def train_one_epoch(self, epoch):
        device = self.cfg['train']['device']
        iters_per_epoch = len(self.dataset) // self.cfg_train['batch_size']
        pbar = tqdm(enumerate(self.dataloader), total=iters_per_epoch, desc='Epoch: {}/{}'.format(epoch, self.cfg_train['total_epochs']))
        for i, data in pbar:
            self.optimG.zero_grad()
            self.realA = data['A'].to(device)
            self.realB = data['B'].to(device)

            ''' 1. Update the two generators '''
            self.update_G()
            ''' 2. Update the two discriminators '''
            self.get_D_predictions()
            self.update_D()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./config.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    cfg_train = cfg['train']
    print(torch.cuda.is_available())
    fix_seeds(3407)
    cycle_gan = CycleGAN(cfg)
    if not os.path.isdir('{}/{}'.format(cfg['output_folder'], cfg_train['model_description'])):
        os.makedirs('{}/{}'.format(cfg['output_folder'], cfg_train['model_description']))
    shutil.copy(args.cfg, '{}/{}'.format(cfg['output_folder'], cfg_train['model_description']))
    epoch_start = 0
    if cfg['train']['keep_train']:
        print('Resume from {}'.format(cfg['train']['path_pretrained_model']))
        epoch_start = cycle_gan.load_model()+1
    for epoch in range(epoch_start, cfg_train['total_epochs']):
        cycle_gan.train_one_epoch(epoch)
        cycle_gan.update_scheduler()
        if (epoch+1) % cfg_train['save_freq'] == 0:
            cycle_gan.save_model(epoch)
        cycle_gan.visualize(epoch)