import logging
import sys
import torch
import torch.nn as nn   
from model import INIT_TYPE, MODEL_NAME
from init_config import *
from train_config import *
from scipy.interpolate import griddata
from matplotlib import gridspec, pyplot as plt
from plot.heatmap import plot_heatmap
import scipy.io

# 打印相关信息
def log(obj):
    print(obj)
    logging.info(obj)
    
if __name__ == "__main__":
    # 设置需要写日志
    init_log()
    # cuda 调用
    device = get_device(sys.argv)
    param_dict = {
        'lb': lb,
        'ub': ub,
        'device': device,
        'path': path,
        'root_path': root_path,
    }


    # 打印参数
    log_str = 'TRAIN_GRID_SIZE '+str(Train_Grid_Size)  + ' Bou_Grid_Size  ' + str(Bou_Grid_Size )
    log(log_str)


    u_region = Exact_u_func(locals()['x_grid'+str(ite_time)],locals()['y_grid'+str(ite_time)])

    train_dict = {
        'x_region': locals()['x_grid'+str(ite_time)],
        'y_region': locals()['y_grid'+str(ite_time)],
        'u_region': u_region,
        'source_region': source(locals()['x_grid'+str(ite_time)],locals()['y_grid'+str(ite_time)]),
        'X_b': X_b,
        'u_b': u_b,
        'u_valid_old':u_valid_old,
        'x_valid': x_valid,
        'y_valid': y_valid,
        'u_valid': u_valid,
        'k': k,
        'ite_time': ite_time,
    }


    ###############################################PINN的模型参数########################################################################
    layer_num = 40
    layers = [2, layer_num, layer_num, layer_num, layer_num, 1]
    model_name = MODEL_NAME.MLP
    model_dict = {
        'layers': layers,
        'init_type': 'default',
        'init_params':{
        },
    }
    log(model_dict)

    
    #train_SOAP(model_dict, model_name, device, param_dict, train_dict, SOAP_steps=200,  SOAP_init_lr=1e-3)

          
    def PlotHeatmap(X,Y, f_res,file_name1 =  '/f_res'):
        X_star = np.hstack((x_grid_MLS, y_grid_MLS))             
        U_star = griddata(X_star, f_res.flatten(), (X_MLS_Grid, Y_MLS_Grid ), method='cubic')
        plot_heatmap(X=X_MLS_Grid, Y=Y_MLS_Grid, Z=U_star, xlabel='x',
                        ylabel='y', file_name=root_path + '/' + TASK_NAME + file_name1)
    
    
    def modelpkl(Z):
        # 加载第Z次模型参数
        net_path = root_path + '/' + path + '/PINN%s.pkl'%(Z)
        model_pkl = PINNConfig.reload_config(net_path=net_path)
        losslist = model_pkl.losslist
        REL2list = model_pkl.REL2list
        RELINFlist = model_pkl.RELINFlist
        scipy.io.savemat(root_path + '/' + TASK_NAME + '/results%s.mat'%(Z), {
        'Loss': losslist,
        'RE_L2': REL2list,
        'RE_LINF': RELINFlist,
        })
        return model_pkl


    def Ite_Results(X,Y,model_pkl):     #希望用这个函数能够告诉我，在进行n次迭代时，前n-1次的Lu在第n次残差点上的值；前n-1次的u在测试点上的值
        X_pred = np.hstack((X, Y))
        x_star = model_pkl.data_loader(X_pred[:, 0:1])
        y_star = model_pkl.data_loader(X_pred[:, 1:2])
        #W = model_pkl.W
        u_pred = model_pkl.net_u(x_star, y_star)
        #print(us_pred.shape)
        #u_pred = torch.matmul(us_pred, W)
        u_pred_x = model_pkl.compute_grad(u_pred,x_star)
        u_pred_xx = model_pkl.compute_grad(u_pred_x,x_star)
        u_pred_xy = model_pkl.compute_grad(u_pred_x,y_star)
        u_pred_y = model_pkl.compute_grad(u_pred,y_star)
        u_pred_yy = model_pkl.compute_grad(u_pred_y,y_star)
        
        #由方程决定
        RHS_new =  u_pred_xx + u_pred_yy + k**2*u_pred
        
        u_pred = model_pkl.detach(u_pred)
        RHS_new = model_pkl.detach(RHS_new)
        u_pred_x = model_pkl.detach(u_pred_x)
        u_pred_xx = model_pkl.detach(u_pred_xx)
        u_pred_y = model_pkl.detach(u_pred_y)
        u_pred_yy = model_pkl.detach(u_pred_yy)
        u_pred_xy = model_pkl.detach(u_pred_xy)
        return u_pred,  u_pred_x, u_pred_y, u_pred_xx,u_pred_xy,  u_pred_yy , RHS_new

    
    def MLS_Sampling(f_res,f_res2,f_res3,Points_number):

        MLS_k= 0.5

        MLS_c= 1

        f_res = np.absolute(f_res)
        f_res2 = np.absolute(f_res2)
        f_res3 = np.absolute(f_res3)

        if f_res2.all() == 0:
            f_res_1_eq = np.power(f_res, MLS_k) / np.mean(np.power(f_res, MLS_k)) 
            f_res_eq = np.power(f_res, MLS_k) / np.mean(np.power(f_res, MLS_k))  + MLS_c

        elif f_res3.all() ==0:
            f_res_1_eq = np.power(f_res, MLS_k) / np.mean(np.power(f_res, MLS_k)) 
            f_res2_eq = np.power(f_res2, MLS_k) / np.mean(np.power(f_res2, MLS_k))
            f_res_eq = 4*f_res_1_eq + 1*f_res2_eq + MLS_c

        else:
            
            f_res_1_eq = np.power(f_res, MLS_k) / np.mean(np.power(f_res, MLS_k)) 
            f_res2_eq = np.power(f_res2, MLS_k) / np.mean(np.power(f_res2, MLS_k))

            f_res3_eq = np.power(f_res3, MLS_k) / np.mean(np.power(f_res3, MLS_k))     

            f_res_eq = 8*f_res_1_eq + 4*f_res2_eq + 1*f_res3_eq +MLS_c                         

        f_res_eq_normalized = (f_res_eq / sum(f_res_eq))[:, 0]
        #print(f_res_eq_normalized.shape)
        MLS_ids = np.random.choice(a=MLS_Grid_Size**2, size=Points_number, replace=False, p=f_res_eq_normalized)
        x_grid_mls = x_grid_MLS[MLS_ids]
        y_grid_mls = y_grid_MLS[MLS_ids]
        #画出采样点
        #设置横坐标刻度
        plt.clf()
        plt.xlim(-1.0, 1.0)
        #设置纵坐标刻度
        plt.ylim(-1.0,1.0)
        plt.gca().set_aspect(1)
        plt.plot(x_grid_mls, y_grid_mls,
                color='red',  # 全部点设置为红色
                marker='.', markersize=0.6, # 点的形状为圆点
                linestyle='', linewidth=1)  # 线型为空，也即点与点之间不用线连接

        file_name0 = root_path + '/' + TASK_NAME+ '/' 
        plt.savefig(file_name0 + 'Iter%s_points_new.JPEG'%ite_time, dpi=500)
        plt.clf()
        return x_grid_mls,y_grid_mls   
    while ite_time<ite:
        
        #第二次迭代
        ite_time += 1
        log('Iter_time: %s' % (ite_time))
        
        #先获得前n-1网络在第n-1次残差点上的源项的值
        u_xx_MLS = 0
        u_yy_MLS = 0
        u_MLS = 0
        u_x_MLS =0
        u_y_MLS =0
        f_res_MLS_sum = 0
        f_res_list = []
        source_MLS = source(x_grid_MLS,y_grid_MLS)
        for i in range(ite_time-1):
            locals()['model_pkl'+str(i+1)] = modelpkl(i+1) 
            model_pkl = locals()['model_pkl'+str(i+1)]            
            locals()['u_MLS'+str(i+1)],locals()['u_x_MLS'+str(i+1)],locals()['u_y_MLS'+str(i+1)],locals()['u_xx_MLS'+str(i+1)],_,locals()['u_yy_MLS'+str(i+1)],_ = Ite_Results(x_grid_MLS,y_grid_MLS,model_pkl)
            u_MLS = u_MLS + locals()['u_MLS'+str(i+1)]
            u_x_MLS = u_x_MLS + locals()['u_x_MLS'+str(i+1)]
            u_y_MLS = u_y_MLS + locals()['u_y_MLS'+str(i+1)]
            u_xx_MLS = u_xx_MLS + locals()['u_xx_MLS'+str(i+1)]
            u_yy_MLS = u_yy_MLS + locals()['u_yy_MLS'+str(i+1)]


            Rx = np.absolute(u_xx_MLS +u_yy_MLS + k**2*u_MLS - source_MLS)
            gamma = 1
            Hx = gamma * (u_x_MLS**2+u_y_MLS**2)**(1/2)
            f_res_list.append(Rx+Hx) 

            
        #MLS 采样
        f_res_MLS_1 = f_res_list[0]
        if ite_time == 2:    
            locals()['x_grid'+str(ite_time)],locals()['y_grid'+str(ite_time)] = MLS_Sampling(f_res_MLS_1,0,0,Train_Grid_Size**2)
        elif ite_time == 3:
            f_res_MLS_2 = f_res_list[1]
            locals()['x_grid'+str(ite_time)],locals()['y_grid'+str(ite_time)] = MLS_Sampling(f_res_MLS_1,f_res_MLS_2,0,Train_Grid_Size**2)
        elif ite_time == 4:
            f_res_MLS_2 = f_res_list[1]
            f_res_MLS_3 = f_res_list[2]
            locals()['x_grid'+str(ite_time)],locals()['y_grid'+str(ite_time)] = MLS_Sampling(f_res_MLS_1,f_res_MLS_2,f_res_MLS_3,Train_Grid_Size**2)                
        

        #下面获得前n-1网络在第n次残差点上的源项的值
        source_new = source(locals()['x_grid'+str(ite_time)],locals()['y_grid'+str(ite_time)])
        #获得第n次训练时前n-1次在边界点上的值
        u_b_new = u_b.copy()
        #获得新的u_valid的值，获得第n次训练时前n-1次在验证点上的值
        u_valid_old_new = u_valid_old
        for i in range(ite_time-1):
            model_pkl = locals()['model_pkl'+str(i+1)]
            _,_,_,_,_,_,locals()['RHS'+str(i+1)] = Ite_Results(locals()['x_grid'+str(ite_time)],locals()['y_grid'+str(ite_time)],model_pkl)
            source_new = source_new - locals()['RHS'+str(i+1)]
            locals()['u_b'+str(i+1)],_,_,_,_,_,_ = Ite_Results(X_b[:, 0:1],X_b[:, 1:2],model_pkl)
            u_b_new = u_b_new - locals()['u_b'+str(i+1)]          
            locals()['u_valid'+str(i+1)],_ ,_,_,_,_,_= Ite_Results(x_valid,y_valid,model_pkl)
            u_valid_old_new = u_valid_old_new + locals()['u_valid'+str(i+1)]  
                    
        log('NewSource_Linf: %e' % np.linalg.norm(source_new,np.inf))
        log('NewSource_L2: %e' % np.linalg.norm(source_new,2))
        
        u_region = Exact_u_func(locals()['x_grid'+str(ite_time)],locals()['y_grid'+str(ite_time)])

        train_dict = {
            'x_region': locals()['x_grid'+str(ite_time)],
            'y_region': locals()['y_grid'+str(ite_time)],
            'u_region': u_region,
            'source_region': source_new,
            'X_b': X_b,
            'u_b': u_b_new,
            'u_valid_old': u_valid_old_new,
            'x_valid': x_valid,
            'y_valid': y_valid,
            'u_valid': u_valid,
            'k': k,
            'ite_time': ite_time,
        }



        # if ite_time == ite:
        #     train_SOAP_SSB(model_dict, model_name, device, param_dict, train_dict, SOAP_steps=0,  SOAP_init_lr=1e-3, SSB_steps=150)
        # elif ite_time == ite-1: 
        #     train_SOAP_SSB(model_dict, model_name, device, param_dict, train_dict, SOAP_steps=200,  SOAP_init_lr=1e-3, SSB_steps=200)

       
       
    def Results_plot(X,Y,Z):
            u_pred_sum = 0
            # source_pred = 0 
            # source_pred1 = -source(X,Y)
            u_pred_sum_list = []
            for i in range(Z):
                model_pkl= modelpkl(i+1)  
                locals()['u_pred'+str(i+1)],_,_,_,_,_,locals()['source_pred'+str(i+2)] = Ite_Results(X,Y,model_pkl)
                u_pred_sum = u_pred_sum  + locals()['u_pred'+str(i+1)]
                u_pred_sum_list.append(u_pred_sum)
                # source_pred = source_pred - locals()['source_pred'+str(i+1)]

                
            X_star = np.hstack((X, Y))
            u_star = Exact_u_func(X_star[:, 0:1], X_star[:, 1:2])
            u_pred = u_pred_sum
            # 计算误差
            RE_Linfinity = np.linalg.norm(u_star-u_pred,np.inf)/np.linalg.norm(u_star,np.inf)
            RE_L2 = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
            log(f'--------FINAL_RE_Linfinity_{Z}iter: {RE_Linfinity:e}--------')
            log(f'------FINAL_RE_L2_{Z}iter: {RE_L2:e}------')

            if len(u_pred_sum_list) > 1:
                log(f'Upred_L2_{Z}iter: {np.linalg.norm(u_pred_sum_list[-1]-u_pred_sum_list[-2],2) / np.linalg.norm(u_pred_sum_list[-1],2):e}-')
                 
                 
            U_star = griddata(X_star, u_star.flatten(), (X_PRED, Y_PRED), method='cubic')
            U_pred = griddata(X_star, u_pred.flatten(), (X_PRED, Y_PRED), method='cubic')
            
            file_name1 = root_path + '/' + TASK_NAME + '/heatmap_true'
            file_name2 = root_path + '/' + TASK_NAME + '/heatmap_pred%s'%(Z)
            file_name3 = root_path + '/' + TASK_NAME + '/heatmap_error%s'%(Z)
            plot_heatmap(X=X_PRED, Y=Y_PRED, Z=U_star, xlabel='x',
                            ylabel='y', file_name=file_name1)
            plot_heatmap(X=X_PRED, Y=Y_PRED, Z=U_pred, xlabel='x',
                            ylabel='y', file_name=file_name2)
            plot_heatmap(X=X_PRED, Y=Y_PRED, Z=np.abs(U_star-U_pred), xlabel='x',
                            ylabel='y', file_name=file_name3)    
            return RE_Linfinity, RE_L2
    for i in range(ite_time):
        RE_Linfinity, RE_L2 = Results_plot(x_valid,y_valid,i+1)