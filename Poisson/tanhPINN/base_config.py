import os
import torch
import numpy as np
from torch import autograd
import logging
from torch.autograd import Variable
from torch import nn
from scipy.optimize import minimize
from pytorch_optimizer import create_optimizer

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=16)

class BaseConfig:
    def __init__(self, lb, ub, device, path, root_path):
        super().__init__()
        # 设置使用设备:cpu, cuda
        self.device = device
        # 上下界
        self.lb = self.data_loader(lb, requires_grad=False)
        self.ub = self.data_loader(ub, requires_grad=False)
        # 当前路径
        self.path = path
        # 根路径
        self.root_path = root_path

        # 训练用的参数
        self.loss = None
        self.optimizer = None
        self.optimizer_name = None
        self.scheduler = None
        self.params = None
        self.init()
        
    def init(self, loss_name='mean', model_name='PINN'):
        '''
        训练参数初始化
        '''
        self.start_time = None
        # 小于这个数是开始保存模型
        self.min_loss = 1e200
        self.min_loss_MMPDE = 1e200
        # 记录运行步数
        self.nIter = 0
        # 损失计算方式
        self.loss_name = loss_name
        self.loss_func_sum = torch.nn.MSELoss(reduction='sum')
        self.loss_func_mean = torch.nn.MSELoss(reduction='mean')
        # 保存模型的名字
        self.model_name = model_name

    # 打印相关信息
    def log(self, obj):
        print(obj)
        logging.info(obj)


    def xavier_init(self, size, requires_grad=True):
        '''
        用xavier_normal随机生成一个可训练的数据
        '''
        W = Variable(nn.init.xavier_normal_(torch.empty(size[0], size[1]))).to(
            self.device)
        W.requires_grad_(requires_grad)
        return W

    def kaiming_init(self, size, requires_grad=True):
        '''
        用kaiming_normal随机生成一个可训练的数据
        '''
        W = Variable(nn.init.kaiming_normal_(torch.empty(size[0], size[1]), mode = 'fan_out', nonlinearity='tanh')).to(
            self.device)
        W.requires_grad_(requires_grad)
        return W

    def data_loader(self, x, requires_grad=True):
        '''
        数据加载函数，转变函数类型及其使用设备设置
        '''
        x_tensor = torch.tensor(x,
                                requires_grad=requires_grad,
                                dtype=torch.float64)
        return x_tensor.to(self.device)

    def reload(self, data, requires_grad=True):
        '''
        重新加载数据
        '''
        data = self.detach(data)
        data = self.data_loader(data, requires_grad=requires_grad)
        return data
    
    def coor_shift(self, X, lb, ub):
        '''
        左边归一化函数，[-1, 1], lb:下确界，ub:上确界
        '''
        X_shift = 2.0 * (X - lb) / (ub - lb) - 1.0
        # X_shift = torch.from_numpy(X_shift).float().requires_grad_()
        return X_shift

    def detach(self, data):
        '''
        将数据从设备上取出
        '''
        tmp_data = data.detach().cpu().numpy()
        if np.isnan(tmp_data).any():
            raise Exception
        return tmp_data

    def reload(self, data, requires_grad=True):
        '''
        重新加载数据
        '''
        data = self.detach(data)
        data = self.data_loader(data, requires_grad=requires_grad)
        return data
    
    def loss_func(self, pred_, true_=None):
        '''
        损失函数计算损失并返回
        '''
        # 采用MSELoss
        if true_ is None:
            true_ = torch.zeros_like(pred_).to(self.device)
            # true_ = self.data_loader(true_)
        if self.loss_name == 'sum':
            return self.loss_func_sum(pred_, true_)
        elif self.loss_name == 'mean':
            return self.loss_func_mean(pred_, true_)
        else:            
            ValueError("The loss_name is not correct!")
            return 0

    def compute_grad(self, u, x):
        '''
        直接计算一阶导数
        '''
        u_x = autograd.grad(u.sum(), x, create_graph=True)[0]
        return u_x
        
    def optimize_one_epoch(self):
        '''
        训练一次
        '''
        return self.loss
    
    def train_SOAP(self, model, SOAP_steps = 50000, SOAP_init_lr = 1e-3):
        '''
        用SOAP训练神经网络
        '''
        SOAP_optimizer = create_optimizer( 
            model,
            'SOAP',
            lr=SOAP_init_lr,
            betas=(0.9, 0.999),
            weight_decay=0,
            eps=1e-8,)

        self.optimizer = SOAP_optimizer
        self.optimizer_name = 'SOAP'
        for it in range(SOAP_steps):
            self.optimize_one_epoch()
            self.optimizer.step()
            
    def train_Adam(self, params, Adam_steps = 50000, Adam_init_lr = 1e-3, scheduler_name=None, scheduler_params=None):
        '''
        用Adam训练神经网络
        '''
        Adam_optimizer = torch.optim.Adam(params=params,
                                        lr=Adam_init_lr,
                                        betas=(0.9, 0.999),
                                        eps=1e-8,
                                        weight_decay=0,
                                      amsgrad=False)
        self.optimizer = Adam_optimizer
        self.optimizer_name = 'Adam'
        if scheduler_name == 'MultiStepLR':
            from torch.optim.lr_scheduler import MultiStepLR
            Adam_scheduler = MultiStepLR(Adam_optimizer, **scheduler_params)
        else:
            Adam_scheduler = None
        self.scheduler = Adam_scheduler
        for it in range(Adam_steps):
            self.optimize_one_epoch()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

    def train_SSB(self, model, device, SSB_steps=10000, method='BFGS', callback=None):
        """
        使用 Scipy 的 optimize.minimize 进行优化。

        参数:
        model: PyTorch 模型
        X: 输入数据
        y: 目标数据
        criterion: 损失函数
        method: Scipy 优化方法，默认为 'BFGS'
        options: Scipy 优化选项，默认为 {'maxiter': 1000}
        callback: 回调函数，用于在每次迭代中获取中间结果

        返回:
        result: Scipy 优化结果
        """
        self.optimizer_name = 'SSB'
        
        # 将模型参数转换为 NumPy 数组
        params = []
        for param in model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        params = np.concatenate(params)

        # 定义目标函数
        def scipy_loss(x):
            # 将 NumPy 参数转换为 PyTorch 张量并加载到模型中
            idx = 0
            for param in model.parameters():
                param.data.copy_(torch.from_numpy(x[idx:idx+param.numel()].reshape(param.shape)).to(device))
                idx += param.numel()
                
            loss =self.optimize_one_epoch()
            gradients = []
            for param in model.parameters():
                gradients.append(param.grad.data.cpu().numpy().flatten())
            gradients = np.concatenate(gradients)

            return loss.item(), gradients        


        H0 = torch.eye(len(params),dtype=torch.float64)
        H0 = H0.numpy()        

        result = minimize(scipy_loss, params, method=method, jac=True, options={'maxiter':SSB_steps, 'gtol': 0, "hess_inv0":H0,
            "method_bfgs":'SSBroyden2', "initial_scale":False})
        

        # 将优化后的参数放回模型中
        idx = 0
        for param in model.parameters():
            param.data.copy_(torch.from_numpy(result.x[idx:idx+param.numel()].reshape(param.shape)).to(device))
            idx += param.numel()


        return result
        
    def train_LBFGS(self, params, LBFGS_steps = 10000, LBFGS_init_lr = 1,  tolerance_LBFGS=-1, LBFGS_scheduler=None):
        LBFGS_optimizer = torch.optim.LBFGS(
            params=params,
            lr=LBFGS_init_lr,
            max_iter=LBFGS_steps,  
            max_eval=LBFGS_steps,
            tolerance_grad=tolerance_LBFGS,
            tolerance_change=tolerance_LBFGS,
            history_size=100,
            line_search_fn='strong_wolfe')
        self.optimizer = LBFGS_optimizer

        self.optimizer_name = 'LBFGS'
        self.scheduler = LBFGS_scheduler

        def closure():
            loss = self.optimize_one_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
            return loss
        try:
            self.optimizer.step(closure)
        except Exception as e:
            print(e)
         
    def train_SGD(self, params, SGD_steps = 50000, SGD_init_lr = 1e-3, scheduler_name=None, scheduler_params=None):
        '''
        用SGD训练神经网络
        '''
        SGD_optimizer = torch.optim.SGD(params=params,
                                        lr=SGD_init_lr)
        self.optimizer = SGD_optimizer
        self.optimizer_name = 'SGD'
        if scheduler_name == 'MultiStepLR':
            from torch.optim.lr_scheduler import MultiStepLR
            SGD_scheduler = MultiStepLR(SGD_optimizer, **scheduler_params)
        else:
            SGD_scheduler = None
        self.scheduler = SGD_scheduler
        for it in range(SGD_steps):
            self.optimize_one_epoch()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

    # @staticmethod
    def save(net, path, name='PINN'):
        '''
        保存模型
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        # 保存神经网络
        torch.save(net, path + '/' + name + '.pkl')  # 保存整个神经网络的结构和模型参数


    @staticmethod
    def reload_config(net_path):
        '''
        载入整个神经网络的结构及其模型参数
        '''
        # 只载入神经网络的模型参数，神经网络的结构需要与保存的神经网络相同的结构
        net = torch.load(net_path, weights_only=False)
        return net

