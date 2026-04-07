import readmsh

import numpy as np
import sys
import matplotlib.pyplot as plt
import QDsim
from math import *
import time
import argparse
import os
import psutil
import config
import Hmatrix as Hmat
from datetime import datetime
# import torch
import Hmatvec_gpu as matvec_gpu

import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"



if __name__ == "__main__":
    fnamegeo='examples/BP5-QD/bp5t.msh'
    fnamePara='examples/BP5-QD/parameter.txt'
    # fnamegeo='examples/WMF/WMF20260201.msh'
    # fnamePara='examples/WMF/parameter.txt'

    
    nodelst,elelst=readmsh.read_mshV2(fnamegeo)
    Para=config.readPara(fnamePara)
    sim0=QDsim.QDsim(elelst,nodelst,Para)
    HMobj=Hmat.Hmatrix(sim0.xg,sim0.nodelst,sim0.elelst,sim0.eleVec,Para)

    Ne=len(sim0.elelst)
    xvector=np.ones(Ne)
    blocks_to_process=HMobj.tree_block
    jud_coredir,blocks_to_process=sim0.get_block_core()
    #blocks_to_process = [blocks_to_process[64],blocks_to_process[65]]
    #print(blocks_to_process)
    macvec = matvec_gpu.BatchedMatVecPreprocessor()
    macvec.transfer_hmatrix(blocks_to_process,Ne)
    
    X_tensor = torch.ones(len(sim0.eleVec), dtype=torch.float64,device='cuda:0')

    std=time.time()
    for i in range(50):
        yvector=macvec.hmatrix_macvec(X_tensor,type='A1d')
    edt=time.time()
    print('gpu matvec time:',edt-std)
    print(yvector)
    
    
    std=time.time()
    k=0
    for k in range(50):
        yvector=np.zeros(len(sim0.eleVec))
        for i in range(len(blocks_to_process)):
            x_=xvector[blocks_to_process[i].col_cluster]

            if(hasattr(blocks_to_process[i], 'judaca') and blocks_to_process[i].judaca==True):
                #Ax_rsvd = blocks_to_process[i].ACA_dictS['U_ACA_Bs'].dot(blocks_to_process[i].ACA_dictS['V_ACA_Bs'].dot(x_))
                #print('!!!!!!!!!!!!!!!!!!!  ',i)
                #k=k+1
                if(len(blocks_to_process[i].ACA_dictD['U_ACA_A1d'])>0):
                    Ax_rsvd = blocks_to_process[i].ACA_dictD['U_ACA_A1d'].dot(blocks_to_process[i].ACA_dictD['V_ACA_A1d'].dot(x_))
                else:
                    Ax_rsvd=np.zeros(len(blocks_to_process[i].row_cluster))
            else:
                #print(blocks_to_process[i].Mf_A1s.shape,len(blocks_to_process[i].row_cluster),len(blocks_to_process[i].col_cluster),x_.shape)
                Ax_rsvd=blocks_to_process[i].Mf_A1d @ x_
                #print(Ax_rsvd)
            yvector[blocks_to_process[i].row_cluster]=yvector[blocks_to_process[i].row_cluster]+Ax_rsvd
            # if(k==2):
            #     break
    print(yvector)
    edt=time.time()
    print('cpu matvec time:',edt-std)

    # plt.figure()
    # plt.plot(yvector)
    # plt.show()
    

    #
    # for i in range(100):  # 模拟 100 次不同的 x
    #     # 生成新的 x_list（每个 x 的维度必须匹配对应矩阵的 N_i）
    #     x_list = [
    #         torch.randn(N, device=macvec.device) 
    #         for N in macvec.N2_list
    #     ]
        
    #     # 核心调用：非常快
    #     y_list = macvec.compute_UVx(x_list)


    