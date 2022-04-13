import torch 
import time
import argparse
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from data.dataset import CycleGANDataset
from torch.utils.data import DataLoader
from itertools import chain
from utils import fix_seeds, history_buffer
from model import *

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

        self.CreateModels(self.cfg_model['num_hourglass'], 
                          self.cfg_model['use_variant'],
                          self.cfg_train['device'])

        self.CreateOptimizers(cfg_train['lr'])

        self.SetScheduler(self.cfg_train['total_epochs'], 
                          self.cfg_train['start_reduce_lr_epoch'])

        self.CreateDatasets(self.cfg_data['domainA_path'], 
                            self.cfg_data['domainB_path'],
                            self.cfg_data['image_size'],
                            self.cfg_data['crop_size'],
                            self.cfg_data['format_domainA'],
                            self.cfg_data['format_domainB'])

        self.CreateDataloaders(self.cfg_train['batch_size'])

        self.CreateLossFunctions()

    def CreateModels(self, num_hourglass, use_variant, device='cuda'):
        self.netG_A2B = Generator(num_hourglass, use_variant).to(device)
        self.netG_B2A = Generator(num_hourglass, use_variant).to(device)
        self.netD_A = Discriminator().to(device)
        self.netD_B = Discriminator().to(device)
    
    def CreateOptimizers(self, lr):
        self.optimG = torch.optim.Adam(chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()), lr=lr, betas=(0.5, 0.999))
        self.optimD = torch.optim.Adam(chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=lr, betas=(0.5, 0.999))

    def SetScheduler(self, total_epochs, start_reduce_lr_epoch):
        lr_lambda = LR_Lambda(total_epochs, start_reduce_lr_epoch)
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimG, lr_lambda=lr_lambda)
        self.lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimD, lr_lambda=lr_lambda)
        
    def CreateDatasets(self, domainA_path, domainB_path, image_size, crop_size, format_domainA = 'png', format_domainB = 'png'):
        self.dataset = CycleGANDataset(domainA_path, format_domainA, domainB_path, format_domainB, image_size, crop_size)

    def CreateDataloaders(self, batch_size=1):
        self.dataloader = DataLoader(self.dataset, batch_size, shuffle=True)

    def CreateLossFunctions(self):
        self.cycle_loss = torch.nn.L1Loss()
        self.identity_loss = torch.nn.L1Loss()
        self.adversarial_loss = torch.nn.MSELoss()

    def train_one_epoch(self, epoch):
        device = self.cfg['train']['device']
        iters_per_epoch = len(self.dataset) // self.cfg_train['batch_size']
        pbar = tqdm(enumerate(self.dataloader), total=iters_per_epoch, desc='Epoch: {}/{}'.format(epoch, self.cfg_train['total_epochs']))
        for i, data in pbar:
            self.optimG.zero_grad()
            self.realA = data['A'].to(device)
            self.realB = data['B'].to(device)

            ''' 1. Update the two generators '''
            self.get_G_predictions()
            self.update_G()
            
            ''' 2. Update the two discriminators '''
            self.get_D_predictions()
            self.update_D()

    def get_G_predictions(self):
        self.fakeB = self.netG_A2B(self.realA)
        self.fakeA = self.netG_B2A(self.realB)
        self.cycleA = self.netG_B2A(self.fakeB)
        self.cycleB = self.netG_A2B(self.fakeA)
        if self.cfg['train']['lambda']['identity'] > 0:
            self.realA2A = self.netG_B2A(self.realA)
            self.realB2B = self.netG_A2B(self.realB)

    def update_G(self):
        ''' 1. Compute losses and do back-propagation on two generators '''
        if self.cfg['train']['lambda']['identity'] < 0:
            identity_loss_A = identity_loss_B = 0.0
        else: 
            identity_loss_A = self.identity_loss(self.realA2A, self.realA) * self.cfg['train']['lambda']['identity']
            identity_loss_B = self.identity_loss(self.realB2B, self.realB) * self.cfg['train']['lambda']['identity']

        cycle_loss_A = self.cycle_loss(self.cycleA, self.realA) * self.cfg['train']['lambda']['cycle']
        cycle_loss_B = self.cycle_loss(self.cycleB, self.realB) * self.cfg['train']['lambda']['cycle'] 
        score_fakeA_from_netD_A = self.netD_A(self.fakeA)
        adversarial_loss_A = self.adversarial_loss(score_fakeA_from_netD_A, torch.ones_like(score_fakeA_from_netD_A))
        score_fakeB_from_netD_B = self.netD_B(self.fakeB)
        adversarial_loss_B = self.adversarial_loss(score_fakeB_from_netD_B, torch.ones_like(score_fakeB_from_netD_B))
        total_G_losses = cycle_loss_A + cycle_loss_B + identity_loss_A + identity_loss_B + adversarial_loss_A + adversarial_loss_B

        ''' 2. Update two generators '''
        self.optimG.zero_grad()
        total_G_losses.backward()
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

    def save_model(self, model_description, epoch, lowest_loss):
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
                'lr_scheduler_D': self.lr_scheduler_D.state_dict(),
            },
            'epoch': epoch,
            'lowest_loss': lowest_loss
        }
        torch.save(check_point, "{}/{}/epoch_{}.pth".format(self.opt['output_folder'], model_description, epoch))

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
    for epoch in range(cfg_train['total_epochs']):
        cycle_gan.train_one_epoch(epoch)
        cycle_gan.update_scheduler()

    # tensor = torch.arange(1, 100, device='cuda')
    # print(torch.cuda.memory_allocated('cuda'))
    # tensor_cpu = tensor.cpu()
    # print(torch.cuda.memory_allocated('cuda'))
    # del tensor
    # print(torch.cuda.memory_allocated('cuda'))
    # tensor1 = torch.arange(1, 100, device='cuda')
    # print(torch.cuda.memory_allocated('cuda'))
    # tensor2 = torch.arange(1, 100, device='cuda')
    # print(torch.cuda.memory_allocated('cuda'))
    # tensor3 = torch.arange(1, 100, device='cuda')
    # print(torch.cuda.memory_allocated('cuda'))
    # tensor4 = torch.arange(1, 100, device='cuda')
    # print(torch.cuda.memory_allocated('cuda'))
    # tensor5 = torch.arange(1, 100, device='cuda')
    # print(torch.cuda.memory_allocated('cuda'))