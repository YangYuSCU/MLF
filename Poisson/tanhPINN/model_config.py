from cmath import inf
import time
import numpy as np
import torch
from base_config import BaseConfig

class PINNConfig(BaseConfig):
    def __init__(self, param_dict, train_dict, model):
        super().__init__(**param_dict)
        self.model = model

        # 加载训练参数
        x_region , y_region,source_region,X_b,u_b,self.u_valid_old,x_valid , y_valid , self.u_valid, self.ite_time = self.unzip_train_dict(
            train_dict=train_dict)

        # 区域内点
        self.x_region = self.data_loader(x_region)
        self.y_region = self.data_loader(y_region)

        self.source_region = self.data_loader(source_region)
        # 边界点
        self.x_b  = self.data_loader(X_b[:, 0:1])
        self.y_b  = self.data_loader(X_b[:, 1:2])
        #边界条件
        self.u_b  = self.data_loader(u_b)
        # 验证
        self.x_valid = self.data_loader(x_valid)
        self.y_valid = self.data_loader(y_valid)


        self.params = list(self.model.parameters())

        self.x_regionshpae = self.x_region.shape[0] 

        self.losslist= []
        self.REL2list = []
        self.RELINFlist = []
        
    def unzip_train_dict(self, train_dict):
        train_data = (
            train_dict['x_region'],
            train_dict['y_region'],
            train_dict['source_region'],
            train_dict['X_b'],
            train_dict['u_b'],
            train_dict['u_valid_old'],
            train_dict['x_valid'],
            train_dict['y_valid'],
            train_dict['u_valid'],
            train_dict['ite_time'],
        )
        return train_data


    def net_u(self, x, y):
        X = torch.cat((x, y), dim=1)
        X = self.coor_shift(X, self.lb, self.ub)
        u = self.model.forward(X)

        return u

    def forward(self, x, y):
        u = self.net_u(x, y)
        return u
    
    
    def Lus(self, u, x, y):
        u_x = self.compute_grad(u,x)
        u_xx = self.compute_grad(u_x , x)
        u_y = self.compute_grad(u, y)
        u_yy = self.compute_grad(u_y, y)
        Lus = -(u_xx + u_yy)
        return Lus
    
    # 训练一次
    def optimize_one_epoch(self):
        if self.start_time is None:
            self.start_time = time.time()
        
        # 初始化loss为0
        self.optimizer.zero_grad()
        self.loss = torch.tensor(0.0, dtype=torch.float64).to(self.device)
        self.loss.requires_grad_()
        # 训练点
        x_region = self.x_region
        y_region = self.y_region
        u_region = self.net_u(x_region,y_region)
        u_region_x = self.compute_grad(u_region, x_region)
        u_region_y = self.compute_grad(u_region, y_region)
        u_region_yy = self.compute_grad(u_region_y, y_region)
        u_region_xx = self.compute_grad(u_region_x, x_region)

        f_u = u_region_xx+ u_region_yy + self.source_region
        
        # 方程损失
        loss_res = self.loss_func(f_u)

        # 边界点
        use_boundary = True
        loss_boundary = torch.tensor(0.0, dtype=torch.float64).to(self.device)
        loss_boundary.requires_grad_()
        if use_boundary:
            x_b = self.x_b
            y_b = self.y_b
            u_b  = self.net_u(x_b,y_b)

            loss_boundary = self.loss_func(u_b, self.u_b)
                           
        # 权重
        alpha_res = 1
        alpha_boundary = 1
        
        self.loss = loss_res * alpha_res + loss_boundary * alpha_boundary
        # 反向传播
        self.loss.backward()
        # 运算次数加1
        self.nIter = self.nIter + 1

        if self.nIter % 10000 == 0:    
            PINNConfig.save(net=self,
                            path=self.root_path + '/' + self.path,
                            name=self.model_name + str(self.ite_time) + '_'+ str(self.nIter))  
        # 保存模型
        loss = self.detach(self.loss)
        if loss < self.min_loss:
            self.min_loss = loss
            PINNConfig.save(net=self,
                            path=self.root_path + '/' + self.path,
                            name=self.model_name+ str(self.ite_time))        

  

         # 打印日志
        loss_remainder = 100
        if np.remainder(self.nIter, loss_remainder) == 0:
            # 打印常规loss
            loss_res = self.detach(loss_res)
            loss_boundary = self.detach(loss_boundary)

            log_str = 'Level ' + str(self.ite_time) + '' + str(self.optimizer_name) + '' + ' Iter ' + str(self.nIter) + ' Loss ' + str(loss) +\
                ' loss_res ' + str(loss_res) + ' loss_boundary ' + str(loss_boundary) +\
                ' LR ' + str(self.optimizer.state_dict()['param_groups'][0]['lr']) +\
                ' min_loss ' + str(self.min_loss)
            self.log(log_str)
            self.losslist.append(loss)
            # 验证
            with torch.no_grad():

                u_valid = self.net_u(self.x_valid,self.y_valid)
                u_valid = self.detach(u_valid)

                L_infinity = np.linalg.norm(self.u_valid - u_valid-self.u_valid_old, np.inf)
                L_2 = np.linalg.norm(self.u_valid - u_valid-self.u_valid_old, ord=2)
                RE_L2 = L_2/np.linalg.norm(self.u_valid, ord=2)
                RE_Linfinity = L_infinity / \
                    np.linalg.norm(self.u_valid, np.inf)

                error_str = ' RE_L2_valid ' + str(RE_L2) + ' RE_Linfinity_valid ' + str(RE_Linfinity) 

                self.log(error_str)
                self.REL2list.append(RE_L2)
                self.RELINFlist.append(RE_Linfinity)
            # 打印耗时
            elapsed = time.time() - self.start_time
            self.log('Time: %.4fs Per %d Iterators' % (elapsed, loss_remainder))
            self.start_time = time.time()          

        return self.loss


