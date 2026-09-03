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


    #画出采样点
    #设置横坐标刻度
    plt.xlim(-0.5, 0.5)
    #设置纵坐标刻度
    plt.ylim(-0.5, 0.5)
    plt.gca().set_aspect(1)
    plt.plot(locals()['x_grid'+str(ite_time)], locals()['y_grid'+str(ite_time)],
        color='red',  # 全部点设置为红色
        marker='.', markersize=0.6, # 点的形状为圆点
        linestyle='', linewidth=1)  # 线型为空，也即点与点之间不用线连接

    file_name0 = root_path + '/' + TASK_NAME+ '/' 
    plt.savefig(file_name0 + 'Iter1_points.JPEG', dpi=500)
    plt.clf()
    train_dict = {
        'x_region': locals()['x_grid'+str(ite_time)],
        'y_region': locals()['y_grid'+str(ite_time)],
        'source_region': source(locals()['x_grid'+str(ite_time)],locals()['y_grid'+str(ite_time)]),
        'X_b': X_b,
        'u_b': u_b,
        'u_valid_old':u_valid_old,
        'x_valid': x_valid,
        'y_valid': y_valid,
        'u_valid': u_valid,
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
    

    #train_SOAP(model_dict, model_name, device, param_dict, train_dict, SOAP_steps=10000,  SOAP_init_lr=1e-3)

    
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

        u_pred = model_pkl.net_u(x_star, y_star)

        u_pred_x = model_pkl.compute_grad(u_pred,x_star)
        u_pred_xx = model_pkl.compute_grad(u_pred_x,x_star)
        u_pred_xy = model_pkl.compute_grad(u_pred_x,y_star)
        u_pred_y = model_pkl.compute_grad(u_pred,y_star)
        u_pred_yy = model_pkl.compute_grad(u_pred_y,y_star)
        
        lap_u_new_tensor = -u_pred_xx - u_pred_yy
        RHS_new =  lap_u_new_tensor
        
        u_pred = model_pkl.detach(u_pred)
        RHS_new = model_pkl.detach(RHS_new)
        u_pred_x = model_pkl.detach(u_pred_x)
        u_pred_xx = model_pkl.detach(u_pred_xx)
        u_pred_y = model_pkl.detach(u_pred_y)
        u_pred_yy = model_pkl.detach(u_pred_yy)
        u_pred_xy = model_pkl.detach(u_pred_xy)
        return u_pred,  u_pred_x, u_pred_y, u_pred_xx,u_pred_xy,  u_pred_yy , RHS_new

 
    def RAD_Sampling(f_res,Points_number):
        #一般来说RAD_k越大越密集
        RAD_k= 0.5
        #一般来说RAD_c越小越密集
        RAD_c= 1 
        #定义残差
        f_res = np.absolute(f_res)

        f_res_eq = np.power(f_res, RAD_k) / np.mean(np.power(f_res, RAD_k)) + RAD_c

        f_res_eq_normalized = (f_res_eq / sum(f_res_eq))[:, 0]

        RAD_ids = np.random.choice(a=RAD_Grid_Size**2, size=Points_number, replace=False, p=f_res_eq_normalized)
        x_grid_rad = x_grid_RAD[RAD_ids]
        y_grid_rad = y_grid_RAD[RAD_ids]
        #画出采样点
        #设置横坐标刻度
        plt.xlim(-0.5, 0.5)
        #设置纵坐标刻度
        plt.ylim(-0.5, 0.5)
        plt.gca().set_aspect(1)
        plt.plot(x_grid_rad, y_grid_rad,
                color='red',  # 全部点设置为红色
                marker='.', markersize=0.6, # 点的形状为圆点
                linestyle='', linewidth=1)  # 线型为空，也即点与点之间不用线连接

        file_name0 = root_path + '/' + TASK_NAME+ '/' 
        plt.savefig(file_name0 + 'Iter%s_points.JPEG'%ite_time, dpi=500)
        plt.clf()
        return x_grid_rad,y_grid_rad 
        
    

    def ML_Sampling(f_res,f_res2,f_res3,Points_number):
        RAD_k= 0.5
        RAD_c= 1

        f_res = np.absolute(f_res)
        f_res2 = np.absolute(f_res2)
        f_res3 = np.absolute(f_res3)
        if f_res2.all() == 0:
            f_res_1_eq = np.power(f_res, RAD_k) / np.mean(np.power(f_res, RAD_k)) 
            f_res_eq = np.power(f_res, RAD_k) / np.mean(np.power(f_res, RAD_k))  + RAD_c
        elif f_res3.all() ==0:
            f_res_1_eq = np.power(f_res, RAD_k) / np.mean(np.power(f_res, RAD_k)) 
            f_res2_eq = np.power(f_res2, RAD_k) / np.mean(np.power(f_res2, RAD_k))
            f_res_eq = 4*f_res_1_eq + 1*f_res2_eq + RAD_c
        else:
            f_res_1_eq = np.power(f_res, RAD_k) / np.mean(np.power(f_res, RAD_k)) 
            f_res2_eq = np.power(f_res2, RAD_k) / np.mean(np.power(f_res2, RAD_k))
            f_res3_eq = np.power(f_res3, RAD_k) / np.mean(np.power(f_res3, RAD_k))     
            f_res_eq = 8*f_res_1_eq + 4*f_res2_eq + 1*f_res3_eq +RAD_c

        f_res_eq_normalized = (f_res_eq / sum(f_res_eq))[:, 0]
        RAD_ids = np.random.choice(a=RAD_Grid_Size**2, size=Points_number, replace=False, p=f_res_eq_normalized)
        x_grid_rad = x_grid_RAD[RAD_ids]
        y_grid_rad = y_grid_RAD[RAD_ids]
        #画出采样点
        #设置横坐标刻度
        plt.clf()
        plt.xlim(-0.5, 0.5)
        #设置纵坐标刻度
        plt.ylim(-0.5,0.5)
        plt.gca().set_aspect(1)
        plt.plot(x_grid_rad, y_grid_rad,
                color='red',  # 全部点设置为红色
                marker='.', markersize=0.6, # 点的形状为圆点
                linestyle='', linewidth=1)  

        file_name0 = root_path + '/' + TASK_NAME+ '/' 
        plt.savefig(file_name0 + 'Iter%s_points.JPEG'%ite_time, dpi=500)
        plt.clf()
        return x_grid_rad,y_grid_rad   

    while ite_time<ite:
        
        #第二次迭代
        ite_time += 1
        log('Iter_time: %s' % (ite_time))
        u_x_RAD = 0
        u_y_RAD = 0        
        u_xx_RAD = 0
        u_yy_RAD = 0
        f_res_RAD_sum = 0
        f_res_list = []
        source_RAD = source(x_grid_RAD,y_grid_RAD)
        for i in range(ite_time-1):
            locals()['model_pkl'+str(i+1)] = modelpkl(i+1) 
            model_pkl = locals()['model_pkl'+str(i+1)]            
            _,locals()['u_x_RAD'+str(i+1)],locals()['u_y_RAD'+str(i+1)],locals()['u_xx_RAD'+str(i+1)],_,locals()['u_yy_RAD'+str(i+1)],_ = Ite_Results(x_grid_RAD,y_grid_RAD,model_pkl)
            u_x_RAD = u_x_RAD + locals()['u_x_RAD'+str(i+1)]
            u_y_RAD = u_y_RAD + locals()['u_y_RAD'+str(i+1)]
            u_xx_RAD = u_xx_RAD + locals()['u_xx_RAD'+str(i+1)]
            u_yy_RAD = u_yy_RAD + locals()['u_yy_RAD'+str(i+1)]
            Rx = np.absolute(-u_xx_RAD -u_yy_RAD - source_RAD)
            gamma = 1
            Hx = gamma * (u_x_RAD**2+u_y_RAD**2)**(1/2)
            f_res_list.append(Rx+Hx) 
            
        f_res_RAD_1 = f_res_list[0]
        
        if ite_time == 2:    
            locals()['x_grid'+str(ite_time)],locals()['y_grid'+str(ite_time)] = ML_Sampling(f_res_RAD_1,0,0,Train_Grid_Size**2)
        elif ite_time == 3:
            f_res_RAD_2 = f_res_list[1]
            locals()['x_grid'+str(ite_time)],locals()['y_grid'+str(ite_time)] = ML_Sampling(f_res_RAD_1,f_res_RAD_2,0,Train_Grid_Size**2)
        elif ite_time == 4:
            f_res_RAD_2 = f_res_list[1]
            f_res_RAD_3 = f_res_list[2]
            locals()['x_grid'+str(ite_time)],locals()['y_grid'+str(ite_time)] = ML_Sampling(f_res_RAD_1,f_res_RAD_2,f_res_RAD_3,Train_Grid_Size**2)                

        #下面获得在第n次残差训练点，利用RAD
        #先获得前n-1网络在第n-1次残差点上的源项的值
        # source_old = source(x_grid_RAD,y_grid_RAD,p)
        # for i in range(ite_time-1):
        #     locals()['model_pkl'+str(i+1)] = modelpkl(i+1) 
        #     model_pkl = locals()['model_pkl'+str(i+1)]            
        #     _,_,_,_,_,_,locals()['RHS'+str(i+1)] = Ite_Results(x_grid_RAD,y_grid_RAD,model_pkl)
        #     source_old = source_old - locals()['RHS'+str(i+1)]
        # locals()['x_grid'+str(ite_time)],locals()['y_grid'+str(ite_time)] = RAD_Sampling(source_old,Train_Grid_Size**2)


        #下面获得前n-1网络在第n次残差点上的源项的值
        source_new = source(locals()['x_grid'+str(ite_time)],locals()['y_grid'+str(ite_time)])
        #获得第n次训练时前n-1次在边界点上的值
        u_b_new = u_b.copy()
        #获得新的u_valid的值，获得第n次训练时前n-1次在验证点上的值
        u_valid_old_new = u_valid_old
        for i in range(ite_time-1):
            #locals()['model_pkl'+str(i+1)] = modelpkl(i+1) 
            model_pkl = locals()['model_pkl'+str(i+1)]
            _,_,_,_,_,_,locals()['RHS'+str(i+1)] = Ite_Results(locals()['x_grid'+str(ite_time)],locals()['y_grid'+str(ite_time)],model_pkl)
            source_new = source_new - locals()['RHS'+str(i+1)]
            locals()['u_b'+str(i+1)],_,_,_,_,_,_ = Ite_Results(X_b[:, 0:1],X_b[:, 1:2],model_pkl)
            u_b_new = u_b_new - locals()['u_b'+str(i+1)]          
            locals()['u_valid'+str(i+1)],_ ,_,_,_,_,_= Ite_Results(x_valid,y_valid,model_pkl)
            u_valid_old_new = u_valid_old_new + locals()['u_valid'+str(i+1)]  
                    
        

        train_dict = {
            'x_region': locals()['x_grid'+str(ite_time)],
            'y_region': locals()['y_grid'+str(ite_time)],
            'source_region': source_new,
            'X_b': X_b,
            'u_b': u_b_new,
            'u_valid_old': u_valid_old_new,
            'x_valid': x_valid,
            'y_valid': y_valid,
            'u_valid': u_valid,
            'ite_time': ite_time,
        }

        #train_SOAP_SSB(model_dict, model_name, device, param_dict, train_dict, SOAP_steps=10000,  SOAP_init_lr=1e-3, SSB_steps=20000)
    
       
       
    def Results_plot(X,Y,Z):
            u_pred_sum = 0

            u_pred_sum_list = []
            for i in range(Z):
                model_pkl= modelpkl(i+1)  
                locals()['u_pred'+str(i+1)],_,_,_,_,_,_ = Ite_Results(X,Y,model_pkl)
                u_pred_sum = u_pred_sum  + locals()['u_pred'+str(i+1)]
                u_pred_sum_list.append(u_pred_sum)

                
            X_star = np.hstack((X, Y))
            u_star = Exact_u_func(X_star[:, 0:1], X_star[:, 1:2])
            u_pred = u_pred_sum
            # 计算误差
            RE_Linfinity = np.linalg.norm(u_star-u_pred,np.inf)/np.linalg.norm(u_star,np.inf)
            RE_L2 = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
            log(f'--------FINAL_RE_Linfinity_{Z}iter: {RE_Linfinity:e}--------')
            log(f'------FINAL_RE_L2_{Z}iter: {RE_L2:e}------')

            if len(u_pred_sum_list) > 1:
                log(f'----Upred_Linf_{Z}iter: {np.linalg.norm(u_pred_sum_list[-1]-u_pred_sum_list[-2],np.inf) / np.linalg.norm(u_pred_sum_list[-1],np.inf):e}----')
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