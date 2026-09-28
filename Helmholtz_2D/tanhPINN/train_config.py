import numpy as np


TASK_NAME = 'Helmholtz2D_Test'

### 训练或者预测可能需要的参数

# 设置定义域
lb = np.array([-1, -1])
ub = np.array([1, 1])

p = 100
k = 6

ite = 3
ite_time = 1



def source(x, y, type='np'):
    if type == 'torch':
        import torch
        pkg = torch
    else:
        pkg = np
    sin = pkg.sin
    cos = pkg.cos

    pi = pkg.pi
    exp = pkg.exp
    tanh = pkg.tanh
    

    u= 4*exp(-p*(x**2+y**2))*sin(k*pi*x)*sin(k*pi*y)
    


    Delta_u = 8*(-pi**2*k**2*sin(pi*k*x)*sin(pi*k*y) - 2*pi*k*p*x*sin(pi*k*y)*cos(pi*k*x) - 2*pi*k*p*y*sin(pi*k*x)*cos(pi*k*y) + 2*p**2*x**2*sin(pi*k*x)*sin(pi*k*y) + 2*p**2*y**2*sin(pi*k*x)*sin(pi*k*y) - 2*p*sin(pi*k*x)*sin(pi*k*y))*exp(-p*(x**2 + y**2))
    
    q = Delta_u + k**2*u
    return q

def Exact_u_func(x, y, type='np'):
    if type == 'torch':
        import torch
        pkg = torch
    else:
        pkg = np
    sin = pkg.sin
    cos = pkg.cos
    pi = pkg.pi
    exp = pkg.exp
    tanh = pkg.tanh

    u= 4*exp(-p*(x**2+y**2))*sin(k*pi*x)*sin(k*pi*y)
    return u

# 区域内部网格大小
Train_Grid_Size = 100
#将整个网格放进去训练

X = np.linspace(lb[0], ub[0], Train_Grid_Size+2)
x = X [1:-1]
Y = np.linspace(lb[1], ub[1], Train_Grid_Size+2)
y = Y [1:-1]

[x_grid,y_grid] = np.meshgrid(x,y)

locals()['x_grid'+str(ite_time)] = x_grid.flatten()[:, None]
locals()['y_grid'+str(ite_time)] = y_grid.flatten()[:, None]
locals()['Jacobi'+str(ite_time)] = 1 


#########################################################################################################
Bou_Grid_Size = 250
Xb = np.linspace(lb[0], ub[0], Bou_Grid_Size+2)
Yb = np.linspace(lb[1], ub[1], Bou_Grid_Size+2)

# 边界点传入
# 外边界网格点
X_Boundary1 = []
X_Boundary2 = []
X_Boundary3 = []
X_Boundary4 = []
for i in Xb:
    for j in Yb:
        if i == lb[0]:
            X_Boundary1.append([i, j]) #左边界
        elif i==ub[0]:
            X_Boundary2.append([i, j]) #右边界
        elif j==lb[1]:
            X_Boundary3.append([i, j]) #下边界
        elif j==ub[1]: 
            X_Boundary4.append([i, j]) #上边界
X_Boundary1 = np.asarray(X_Boundary1, dtype=float)
X_Boundary2 = np.asarray(X_Boundary2, dtype=float)
X_Boundary3 = np.asarray(X_Boundary3, dtype=float)
X_Boundary4 = np.asarray(X_Boundary4, dtype=float)

#采样全部边界点
X_b_left  = X_Boundary1
X_b_right = X_Boundary2
X_b_down = X_Boundary3
X_b_up = X_Boundary4

u_b_left = Exact_u_func(X_b_left[:, 0:1],X_b_left[:, 1:2])
u_b_right = Exact_u_func(X_b_right[:, 0:1],X_b_right[:, 1:2])
u_b_down = Exact_u_func(X_b_down[:, 0:1],X_b_down[:, 1:2])
u_b_up = Exact_u_func(X_b_up[:, 0:1],X_b_up[:, 1:2])  


X_b = np.concatenate((X_b_left, X_b_right,X_b_down,X_b_up))
u_b = np.concatenate((u_b_left, u_b_right,u_b_down,u_b_up))


PRED_GRID_SIZE = 399
X_PRED_GRID = np.linspace(lb[0], ub[0], PRED_GRID_SIZE)
Y_PRED_GRID = np.linspace(lb[1], ub[1], PRED_GRID_SIZE)
X_PRED, Y_PRED = np.meshgrid(X_PRED_GRID, Y_PRED_GRID)
x_valid = X_PRED.flatten()[:, None]
y_valid = Y_PRED.flatten()[:, None]
u_valid = Exact_u_func(x_valid,y_valid)

#生成预备加入采点的数据
MLS_Grid_Size = 500


X_MLS = np.linspace(lb[0], ub[0], MLS_Grid_Size+2)
x_MLS = X_MLS[1:-1]
Y_MLS = np.linspace(lb[1], ub[1], MLS_Grid_Size+2)
y_MLS = Y_MLS[1:-1]

X_MLS_Grid, Y_MLS_Grid = np.meshgrid(x_MLS, y_MLS)

x_grid_MLS = X_MLS_Grid.flatten()[:, None]
y_grid_MLS = Y_MLS_Grid.flatten()[:, None]


u_valid_old = 0