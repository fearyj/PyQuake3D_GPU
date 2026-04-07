import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from config import comm, rank, size

class BatchedMatVecPreprocessor:
    """
    预处理 A_list,只执行一次昂贵的 flatten / cat / offsets 计算。
    后续只需传入新的 x_list 即可快速计算 y_list。
    """
    def __init__(self,GPU_cores=1):
        # torch.set_default_device('cuda')
        # self.device='cuda'
        #torch.set_default_device('cpu')
        #self.device='cpu'
        #self.device=f'cuda:{rank}'
        #self.device=f'cuda:0' if torch.cuda.is_available() else 'cpu'
        grank=int(rank%GPU_cores)
        self.blocksize = (50, 50)
        self.device=f'cuda:{grank}' if torch.cuda.is_available() else 'cpu'
        pass

    def transfer_hmatrix(self,hamtrix_lst,Ne):
        print('start to transfer Hamtrix_lst to torch tensor list.')
        Um_lst_A1s=[]
        Vt_lst_A1s=[]
        Um_lst_A2s=[]
        Vt_lst_A2s=[]
        Um_lst_Bs=[]
        Vt_lst_Bs=[]
        Um_lst_A1d=[]
        Vt_lst_A1d=[]
        Um_lst_A2d=[]
        Vt_lst_A2d=[]
        Um_lst_Bd=[]
        Vt_lst_Bd=[]

        Am_lst_A1s=[]
        Am_lst_A2s=[]
        Am_lst_Bs=[]
        Am_lst_A1d=[]
        Am_lst_A2d=[]
        Am_lst_Bd=[]

        self.dices_uv_col_lst=[]
        self.dices_uv_row_lst=[]
        self.dices_m_col_lst=[]
        self.dices_m_row_lst=[]
        self.Ne=Ne
        
        std=time.time()
        for i in range(len(hamtrix_lst)):
            if(hasattr(hamtrix_lst[i], 'judaca') and hamtrix_lst[i].judaca==True):
                Um = hamtrix_lst[i].ACA_dictS['U_ACA_A1s']
                Vt=hamtrix_lst[i].ACA_dictS['V_ACA_A1s']
                #print(Um.shape,Vt.shape)
                Um_lst_A1s.append(torch.from_numpy(Um).double())
                Vt_lst_A1s.append(torch.from_numpy(Vt).double())

                Um = hamtrix_lst[i].ACA_dictS['U_ACA_A2s']
                Vt=hamtrix_lst[i].ACA_dictS['V_ACA_A2s']
                #print(hamtrix_lst[i].ACA_dictS['V_ACA_A1s'].shape,hamtrix_lst[i].ACA_dictS['V_ACA_A2s'].shape)
                Um_lst_A2s.append(torch.from_numpy(Um).double())
                Vt_lst_A2s.append(torch.from_numpy(Vt).double())

                Um = hamtrix_lst[i].ACA_dictS['U_ACA_Bs']
                Vt=hamtrix_lst[i].ACA_dictS['V_ACA_Bs']
                #print(Um.shape,Vt.shape)
                if(len(Um)>0):
                    
                    Um_lst_Bs.append(torch.from_numpy(Um).double())
                    Vt_lst_Bs.append(torch.from_numpy(Vt).double())


                Um = hamtrix_lst[i].ACA_dictD['U_ACA_A1d']
                Vt=hamtrix_lst[i].ACA_dictD['V_ACA_A1d']
                #print(Um.shape,Vt.shape)
                Um_lst_A1d.append(torch.from_numpy(Um).double())
                Vt_lst_A1d.append(torch.from_numpy(Vt).double())

                Um = hamtrix_lst[i].ACA_dictD['U_ACA_A2d']
                Vt=hamtrix_lst[i].ACA_dictD['V_ACA_A2d']
                #print(Um.shape,Vt.shape)
                Um_lst_A2d.append(torch.from_numpy(Um).double())
                Vt_lst_A2d.append(torch.from_numpy(Vt).double())

                Um = hamtrix_lst[i].ACA_dictD['U_ACA_Bd']
                Vt=hamtrix_lst[i].ACA_dictD['V_ACA_Bd']
                #print(Um.shape,Vt.shape)
                if(len(Um)>0):
                    Um_lst_Bd.append(torch.from_numpy(Um).double())
                    Vt_lst_Bd.append(torch.from_numpy(Vt).double())



                self.dices_uv_col_lst.append(hamtrix_lst[i].col_cluster)
                self.dices_uv_row_lst.append(hamtrix_lst[i].row_cluster)
            else:
                #print(blocks_to_process[i].Mf_A1s.shape,len(blocks_to_process[i].row_cluster),len(blocks_to_process[i].col_cluster),x_.shape)
                Am=hamtrix_lst[i].Mf_A1s
                Am_lst_A1s.append(torch.from_numpy(Am).double())
                Am=hamtrix_lst[i].Mf_A2s
                Am_lst_A2s.append(torch.from_numpy(Am).double())
                Am=hamtrix_lst[i].Mf_Bs
                Am_lst_Bs.append(torch.from_numpy(Am).double())
                Am=hamtrix_lst[i].Mf_A1d
                Am_lst_A1d.append(torch.from_numpy(Am).double())
                Am=hamtrix_lst[i].Mf_A2d
                Am_lst_A2d.append(torch.from_numpy(Am).double())
                Am=hamtrix_lst[i].Mf_Bd
                Am_lst_Bd.append(torch.from_numpy(Am).double())

                R, C = np.meshgrid(hamtrix_lst[i].row_cluster, hamtrix_lst[i].col_cluster, indexing='ij') 
                self.dices_m_col_lst.append(C)
                self.dices_m_row_lst.append(R)
                #print(R.shape,C.shape,hamtrix_lst[i].Mf_A1s.shape)
                # self.dices_m_col_lst.append(hamtrix_lst[i].col_cluster)
                # self.dices_m_row_lst.append(hamtrix_lst[i].row_cluster)
                #print(hamtrix_lst[i].Mf_A1s.shape,hamtrix_lst[i].col_cluster.shape,hamtrix_lst[i].row_cluster.shape)
        
        self.N1_list = [U.shape[0] for U in Um_lst_A1s]      # 每个 block 的输出维度（U 的行数）
        self.N2_list = [V.shape[1] for V in Vt_lst_A1s]      # 每个 block 的输入维度（V 的列数）
        self.R_list_1S  = [U.shape[1] for U in Um_lst_A1s]      # 每个 block 的秩 R
        self.R_list_2S  = [U.shape[1] for U in Um_lst_A2s]      # 每个 block 的秩 R
        self.R_list_1D  = [U.shape[1] for U in Um_lst_A1d]      # 每个 block 的秩 R
        self.R_list_2D  = [U.shape[1] for U in Um_lst_A2d]      # 每个 block 的秩 R
        
        self.V_segment_length_1S = np.repeat(
            np.asarray(self.N2_list, dtype=np.int64),
            np.asarray(self.R_list_1S, dtype=np.int64)
        )
        self.V_segment_length_2S = np.repeat(
            np.asarray(self.N2_list, dtype=np.int64),
            np.asarray(self.R_list_2S, dtype=np.int64)
        )

        self.V_segment_length_1D = np.repeat(
            np.asarray(self.N2_list, dtype=np.int64),
            np.asarray(self.R_list_1D, dtype=np.int64)
        )
        self.V_segment_length_2D = np.repeat(
            np.asarray(self.N2_list, dtype=np.int64),
            np.asarray(self.R_list_2D, dtype=np.int64)
        )

        self.U_segment_length_1S = np.repeat(
            np.asarray(self.R_list_1S, dtype=np.int64),
            np.asarray(self.N1_list, dtype=np.int64)
        )
        self.U_segment_length_2S = np.repeat(
            np.asarray(self.R_list_2S, dtype=np.int64),
            np.asarray(self.N1_list, dtype=np.int64)
        )

        self.U_segment_length_1D = np.repeat(
            np.asarray(self.R_list_1D, dtype=np.int64),
            np.asarray(self.N1_list, dtype=np.int64)
        )
        self.U_segment_length_2D = np.repeat(
            np.asarray(self.R_list_2D, dtype=np.int64),
            np.asarray(self.N1_list, dtype=np.int64)
        )
        self.Bs_jud=True
        self.Bd_jud=True
        self.Um_flat_A1s,self.Vt_flat_A1s=self.init_UV(Um_lst_A1s, Vt_lst_A1s)
        self.Um_flat_A2s,self.Vt_flat_A2s=self.init_UV(Um_lst_A2s, Vt_lst_A2s)
        if(len(Um_lst_Bs)>0):
            self.Um_flat_Bs,self.Vt_flat_Bs=self.init_UV(Um_lst_Bs, Vt_lst_Bs)
        else:
            self.Bs_jud=False
        

        self.Um_flat_A1d,self.Vt_flat_A1d=self.init_UV(Um_lst_A1d, Vt_lst_A1d)
        self.Um_flat_A2d,self.Vt_flat_A2d=self.init_UV(Um_lst_A2d, Vt_lst_A2d)
        if(len(Um_lst_Bd)>0):
            self.Um_flat_Bd,self.Vt_flat_Bd=self.init_UV(Um_lst_Bd, Vt_lst_Bd)
        else:
            self.Bd_jud=False
        self.V_total_elements_1S = self.Vt_flat_A1s.numel()
        self.U_total_elements_1S = self.Um_flat_A1s.numel()
        self.V_total_elements_2S = self.Vt_flat_A2s.numel()
        self.U_total_elements_2S = self.Um_flat_A2s.numel()
        self.V_total_elements_1D = self.Vt_flat_A1d.numel()
        self.U_total_elements_1D = self.Um_flat_A1d.numel()
        self.V_total_elements_2D = self.Vt_flat_A2d.numel()
        self.U_total_elements_2D = self.Um_flat_A2d.numel()


        #self.init_UV(U_list=Um_lst_A1s,V_list=Vt_lst_A1s,device='cuda' if torch.cuda.is_available() else 'cpu')
        self.init_m_sparse(Am_lst_A1s,Am_lst_A2s,Am_lst_Bs,Am_lst_A1d,Am_lst_A2d,Am_lst_Bd,N1=Ne,N2=Ne)
        
        # endt=time.time()
        # print('time init:',endt-std)

        self.transVX_in_csrS1=self.init_trans_vx_in_matrix(N1=self.V_total_elements_1S,N2=Ne,R_list=self.R_list_1S)
        self.transVX_in_csrD1=self.init_trans_vx_in_matrix(N1=self.V_total_elements_1D,N2=Ne,R_list=self.R_list_1D)
        
        self.transVX_in_csrS2=self.init_trans_vx_in_matrix(N1=self.V_total_elements_2S,N2=Ne,R_list=self.R_list_2S)
        self.transVX_in_csrD2=self.init_trans_vx_in_matrix(N1=self.V_total_elements_2D,N2=Ne,R_list=self.R_list_2D)
        
        # endt=time.time()
        # print('time init:',endt-std)
        self.transVX_out_csrS1=self.init_trans_vx_out_matrix(N1=len(self.V_segment_length_1S),N2=self.V_total_elements_1S,V_segment_lengths=self.V_segment_length_1S)
        self.transVX_out_csrD1=self.init_trans_vx_out_matrix(N1=len(self.V_segment_length_1D),N2=self.V_total_elements_1D,V_segment_lengths=self.V_segment_length_1D)

        self.transVX_out_csrS2=self.init_trans_vx_out_matrix(N1=len(self.V_segment_length_2S),N2=self.V_total_elements_2S,V_segment_lengths=self.V_segment_length_2S)
        self.transVX_out_csrD2=self.init_trans_vx_out_matrix(N1=len(self.V_segment_length_2D),N2=self.V_total_elements_2D,V_segment_lengths=self.V_segment_length_2D)
        #torch.cuda.empty_cache()
        # endt=time.time()
        # print('time init:',endt-std)
        self.transUX_in_csrS1=self.init_trans_ux_in_matrix(N1=self.U_total_elements_1S,N2=len(self.V_segment_length_1S),N1_list=self.N1_list,R_list=self.R_list_1S)
        self.transUX_in_csrD1=self.init_trans_ux_in_matrix(N1=self.U_total_elements_1D,N2=len(self.V_segment_length_1D),N1_list=self.N1_list,R_list=self.R_list_1D)

        self.transUX_in_csrS2=self.init_trans_ux_in_matrix(N1=self.U_total_elements_2S,N2=len(self.V_segment_length_2S),N1_list=self.N1_list,R_list=self.R_list_2S)
        self.transUX_in_csrD2=self.init_trans_ux_in_matrix(N1=self.U_total_elements_2D,N2=len(self.V_segment_length_2D),N1_list=self.N1_list,R_list=self.R_list_2D)
        #torch.cuda.empty_cache()
        # endt=time.time()
        # print('time init:',endt-std)
        self.transUX_out_csrS1=self.init_trans_ux_out_matrix(N1=len(self.U_segment_length_1S),N2=self.U_total_elements_1S,U_total_elements=self.U_total_elements_1S,U_segment_lengths=self.U_segment_length_1S)        
        self.transUX_out_csrD1=self.init_trans_ux_out_matrix(N1=len(self.U_segment_length_1D),N2=self.U_total_elements_1D,U_total_elements=self.U_total_elements_1D,U_segment_lengths=self.U_segment_length_1D)        
        
        self.transUX_out_csrS2=self.init_trans_ux_out_matrix(N1=len(self.U_segment_length_2S),N2=self.U_total_elements_2S,U_total_elements=self.U_total_elements_2S,U_segment_lengths=self.U_segment_length_2S)        
        self.transUX_out_csrD2=self.init_trans_ux_out_matrix(N1=len(self.U_segment_length_2D),N2=self.U_total_elements_2D,U_total_elements=self.U_total_elements_2D,U_segment_lengths=self.U_segment_length_2D)        
        #torch.cuda.empty_cache()
        # endt=time.time()
        # print('time init:',endt-std)

        self.trans_y_csrS1=self.init_trans_ux_y_matrix(N1=Ne,N2=len(self.U_segment_length_1S),U_segment_lengths=self.U_segment_length_1S,N1_list=self.N1_list)
        self.trans_y_csrD1=self.init_trans_ux_y_matrix(N1=Ne,N2=len(self.U_segment_length_1D),U_segment_lengths=self.U_segment_length_1D,N1_list=self.N1_list)
        
        self.trans_y_csrS2=self.init_trans_ux_y_matrix(N1=Ne,N2=len(self.U_segment_length_2S),U_segment_lengths=self.U_segment_length_2S,N1_list=self.N1_list)
        self.trans_y_csrD2=self.init_trans_ux_y_matrix(N1=Ne,N2=len(self.U_segment_length_2D),U_segment_lengths=self.U_segment_length_2D,N1_list=self.N1_list)
        
        
        endt=time.time()
        print('time init:',endt-std)
        # endt=time.time()
        # print('time init:',endt-std)
        # self.init_trans_vx_out_matrix(N1=len(self.V_segment_lengths),N2=self.V_total_elements)
        # self.init_trans_ux_in_matrix(N1=self.U_total_elements,N2=len(self.V_segment_lengths))
        # endt=time.time()
        # print('time init:',endt-std)
        # self.init_trans_ux_out_matrix(N1=len(self.U_segment_lengths),N2=self.U_total_elements)
        # self.init_trans_ux_y_matrix(N1=Ne,N2=len(self.U_segment_lengths))
        # endt=time.time()
        # print('time init:',endt-std)

    def init_m_sparse(self,Am_A1s,Am_A2s,Am_Bs,Am_A1d,Am_A2d,Am_Bd,N1,N2):
        # dices_m_row_lst=np.concatenate(self.dices_m_row_lst)
        # dices_m_col_lst=np.concatenate(self.dices_m_col_lst)
        
        row_indices = np.concatenate([A.flatten() for A in self.dices_m_row_lst])
        col_indices = np.concatenate([A.flatten() for A in self.dices_m_col_lst])

        row_indices = torch.from_numpy(row_indices).to(self.device)
        col_indices = torch.from_numpy(col_indices).to(self.device)
        m_flats = [A.flatten() for A in Am_A1s]
        m_flat = torch.cat(m_flats).to(self.device)
        #print(row_indices.shape,col_indices.shape,self.m_flat.shape)
        coo = torch.sparse_coo_tensor(
                indices=torch.stack([row_indices, col_indices]),
                values=m_flat,
                size=(N1,N2)
            )
        self.trans_M_csr_A1s =coo.to_sparse_csr()
        

        m_flats = [A.flatten() for A in Am_A2s]
        m_flat = torch.cat(m_flats).to(self.device)
        #print(row_indices.shape,col_indices.shape,self.m_flat.shape)
        coo = torch.sparse_coo_tensor(
                indices=torch.stack([row_indices, col_indices]),
                values=m_flat,
                size=(N1,N2)
            )
        self.trans_M_csr_A2s =coo.to_sparse_csr()

        m_flats = [A.flatten() for A in Am_Bs]
        m_flat = torch.cat(m_flats).to(self.device)
        #print(row_indices.shape,col_indices.shape,self.m_flat.shape)
        coo = torch.sparse_coo_tensor(
                indices=torch.stack([row_indices, col_indices]),
                values=m_flat,
                size=(N1,N2)
            )
        self.trans_M_csr_Bs =coo.to_sparse_csr()

        m_flats = [A.flatten() for A in Am_A1d]
        m_flat = torch.cat(m_flats).to(self.device)
        #print(row_indices.shape,col_indices.shape,self.m_flat.shape)
        coo = torch.sparse_coo_tensor(
                indices=torch.stack([row_indices, col_indices]),
                values=m_flat,
                size=(N1,N2)
            )

        self.trans_M_csr_A1d =coo.to_sparse_csr()

        m_flats = [A.flatten() for A in Am_A2d]
        m_flat = torch.cat(m_flats).to(self.device)
        #print(row_indices.shape,col_indices.shape,self.m_flat.shape)
        coo = torch.sparse_coo_tensor(
                indices=torch.stack([row_indices, col_indices]),
                values=m_flat,
                size=(N1,N2)
            )
        self.trans_M_csr_A2d =coo.to_sparse_csr()

        m_flats = [A.flatten() for A in Am_Bd]
        m_flat = torch.cat(m_flats).to(self.device)
        #print(row_indices.shape,col_indices.shape,self.m_flat.shape)
        coo = torch.sparse_coo_tensor(
                indices=torch.stack([row_indices, col_indices]),
                values=m_flat,
                size=(N1,N2)
            )
        self.trans_M_csr_Bd =coo.to_sparse_csr()
        del coo




    def init_trans_vx_in_matrix(self,N1,N2,R_list):
        row_indices = []
        col_indices  = []
        Srow=0
        values= np.ones(N1)
        for i in range(len(self.dices_uv_col_lst)):
            for j in range(R_list[i]):
                col_indices.append(self.dices_uv_col_lst[i])  #only one none-zero index in each row, from col_lst,to enlarge the X vec
                
                # for k in range(len(self.dices_uv_col_lst[i])): 
                #     row_indices.append(Srow)   #place the repeated X vector in order to correspond to the flattened V.
                #     Srow=Srow+1
        row_indices=np.arange(0,N1,1)
        #row_indices=np.array(row_indices)
        col_indices=np.concatenate(col_indices)
        
        row_indices = torch.from_numpy(row_indices).to(self.device)
        col_indices = torch.from_numpy(col_indices).to(self.device)
        row_indices = row_indices.to(torch.int32)
        col_indices = col_indices.to(torch.int32)
        values = torch.from_numpy(values).to(self.device)
        #transX_csr = torch.sparse_csr_tensor(row_indices, col_indices, values, size=(N1,N2))
        coo = torch.sparse_coo_tensor(
                indices=torch.stack([row_indices, col_indices]),
                values=values,
                size=(N1,N2)
            )
        transVX_in_csr =coo.to_sparse_csr()
        del coo
        return transVX_in_csr





    def init_trans_vx_out_matrix(self,N1,N2,V_segment_lengths):
        row_indices = []
        col_indices  = []
        Scol=0
        Srow=0
        values= np.ones(N2)
        
        # for i in range(len(self.V_segment_lengths)):
        #     for j in range(self.V_segment_lengths[i]):
        #         row_indices.append(Srow) #sum the values for each V_segment_lengths as the output each row
        #         col_indices.append(Scol)
        #         # if(i<30):
        #         #     print(j,Scol)
        #         Scol=Scol+1
        #     Srow=Srow+1
        # row_indices=np.array(row_indices)
        # col_indices=np.array(col_indices)

        # for seg_len in self.V_segment_lengths:
        #     row_indices.append([Srow] * seg_len)
        #     col_indices.append(range(Scol, Scol + seg_len))
        #     Scol += seg_len
        #     Srow += 1
        # row_indices=np.concatenate(row_indices)
        # col_indices=np.concatenate(col_indices)

        row_indices = np.repeat(
            np.arange(len(V_segment_lengths)),
            V_segment_lengths
        )
        col_indices = np.arange(np.sum(V_segment_lengths))
        
        row_indices = torch.from_numpy(row_indices).to(self.device)
        col_indices = torch.from_numpy(col_indices).to(self.device)
        row_indices = row_indices.to(torch.int32)
        col_indices = col_indices.to(torch.int32)
        values = torch.from_numpy(values).to(self.device)
        #transVX_csr = torch.sparse_csr_tensor(row_indices, col_indices, values, size=(N1,N2))
        coo = torch.sparse_coo_tensor(
                indices=torch.stack([row_indices, col_indices]),
                values=values,
                size=(N1,N2)
            )
        transVX_out_csr =coo.to_sparse_csr()
        del coo
        return transVX_out_csr

    
    def init_trans_ux_in_matrix(self,N1,N2,N1_list,R_list):
        std=time.time()
        row_indices = []
        col_indices  = []
        Scol=0
        Srow=0
        values= np.ones(N1)

        
        # for N_, R_ in zip(N1_list, R_list):
        #     col_indices.extend(
        #         [c for _ in range(N_) for c in range(Scol, Scol + R_)]
        #     )
        #     Scol += R_
        # row_indices=np.arange(N1)
        # col_indices=np.array(col_indices)

        
        col_indices = np.empty(N1)  # 或 int64 根据需求
        row_indices=np.arange(N1)
        
        pos = 0
        current_col = Scol
        
        for N_, R_ in zip(N1_list, R_list):
            block = np.arange(current_col, current_col + R_, dtype=col_indices.dtype)
            # tile 比 repeat 通常更快一点（内存布局友好）
            col_indices[pos : pos + N_*R_] = np.tile(block, N_)
            pos += N_ * R_
            current_col += R_


        row_indices = torch.from_numpy(row_indices).to(self.device)
        col_indices = torch.from_numpy(col_indices).to(self.device)
        row_indices = row_indices.to(torch.int32)
        col_indices = col_indices.to(torch.int32)
        values = torch.from_numpy(values).to(self.device)
        #transX_csr = torch.sparse_csr_tensor(row_indices, col_indices, values, size=(N1,N2))
        coo = torch.sparse_coo_tensor(
                indices=torch.stack([row_indices, col_indices]),
                values=values,
                size=(N1,N2)
            )
        transUX_in_csr =coo.to_sparse_csr()
        del coo
        return transUX_in_csr

    
    def init_trans_ux_out_matrix(self,N1,N2,U_total_elements,U_segment_lengths):
        row_indices = []
        col_indices  = []

        values= np.ones(U_total_elements)
        Nall=np.sum(U_segment_lengths)

        row_indices = np.repeat(
            np.arange(len(U_segment_lengths)),
            U_segment_lengths
        )
        col_indices = np.arange(Nall)


        row_indices = torch.from_numpy(row_indices).to(self.device)
        col_indices = torch.from_numpy(col_indices).to(self.device)
        row_indices = row_indices.to(torch.int32)
        col_indices = col_indices.to(torch.int32)
        values = torch.from_numpy(values).to(self.device)
        #transVX_csr = torch.sparse_csr_tensor(row_indices, col_indices, values, size=(N1,N2))
        #print(row_indices.dtype,col_indices.dtype)
        coo = torch.sparse_coo_tensor(
                indices=torch.stack([row_indices, col_indices]),
                values=values,
                size=(N1,N2)
            )
        transUX_out_csr =coo.to_sparse_csr()
        del coo
        return transUX_out_csr

    def init_trans_ux_y_matrix(self,N1,N2,U_segment_lengths,N1_list):
        row_indices = []
        col_indices  = []
        Scol=0
        Srow=0
        values= np.ones(len(U_segment_lengths))
        
        for i in range(len(N1_list)):
            row_indices.append(self.dices_uv_row_lst[i])
            # for k in range(len(self.dices_uv_row_lst[i])): 
            #     col_indices.append(Scol)   #place the repeated X vector in order to correspond to the flattened V.
            #     Scol=Scol+1
        
        row_indices=np.concatenate(row_indices)
        col_indices=np.arange(len(U_segment_lengths))
        
        row_indices = torch.from_numpy(row_indices).to(self.device)
        col_indices = torch.from_numpy(col_indices).to(self.device)
        row_indices = row_indices.to(torch.int32)
        col_indices = col_indices.to(torch.int32)
        values = torch.from_numpy(values).to(self.device)
        #transX_csr = torch.sparse_csr_tensor(row_indices, col_indices, values, size=(N1,N2))
        coo = torch.sparse_coo_tensor(
                indices=torch.stack([row_indices, col_indices]),
                values=values,
                size=(N1,N2)
            )
        trans_y_csr =coo.to_sparse_csr()
        del coo
        return trans_y_csr

        

    
    def hmatrix_macvec(self,X,type):
        #std=time.time()
        X = X.to(dtype=torch.float64, device=self.device)
        if(type=='A1s'):
            V_flat=self.Vt_flat_A1s
            U_flat=self.Um_flat_A1s
            trans_M_csr=self.trans_M_csr_A1s
        if(type=='A2s'):
            V_flat=self.Vt_flat_A2s
            U_flat=self.Um_flat_A2s
            trans_M_csr=self.trans_M_csr_A2s
        if(type=='Bs'):
            if(self.Bs_jud==False):
                y_uv= torch.zeros(self.Ne, dtype=torch.float64,device=self.device)
            else:        
                V_flat=self.Vt_flat_Bs
                U_flat=self.Um_flat_Bs
            trans_M_csr=self.trans_M_csr_Bs
            
        if(type=='A1d'):
            V_flat=self.Vt_flat_A1d
            U_flat=self.Um_flat_A1d
            trans_M_csr=self.trans_M_csr_A1d
        if(type=='A2d'):
            V_flat=self.Vt_flat_A2d
            U_flat=self.Um_flat_A2d
            trans_M_csr=self.trans_M_csr_A2d
        if(type=='Bd'):
            if(self.Bd_jud==False):
                y_uv= torch.zeros(self.Ne, dtype=torch.float64,device=self.device)
            else:
                V_flat=self.Vt_flat_Bd
                U_flat=self.Um_flat_Bd
            trans_M_csr=self.trans_M_csr_Bd
        
        
        if(type=='A1s' or type=='Bs'):
            transVX_in_csr=self.transVX_in_csrS1
            transVX_out_csr=self.transVX_out_csrS1
            transUX_in_csr=self.transUX_in_csrS1
            transUX_out_csr=self.transUX_out_csrS1
            trans_y_csr=self.trans_y_csrS1
        if(type=='A2s'):
            transVX_in_csr=self.transVX_in_csrS2
            transVX_out_csr=self.transVX_out_csrS2
            transUX_in_csr=self.transUX_in_csrS2
            transUX_out_csr=self.transUX_out_csrS2
            trans_y_csr=self.trans_y_csrS2
        if(type=='A1d' or type=='Bd'):
            transVX_in_csr=self.transVX_in_csrD1
            transVX_out_csr=self.transVX_out_csrD1
            transUX_in_csr=self.transUX_in_csrD1
            transUX_out_csr=self.transUX_out_csrD1
            trans_y_csr=self.trans_y_csrD1
        if(type=='A2d'):
            transVX_in_csr=self.transVX_in_csrD2
            transVX_out_csr=self.transVX_out_csrD2
            transUX_in_csr=self.transUX_in_csrD2
            transUX_out_csr=self.transUX_out_csrD2
            trans_y_csr=self.trans_y_csrD2
        

        if((type=='Bs' and self.Bs_jud==False) or (type=='Bd' and self.Bd_jud==False)):
            #print(trans_M_csr.device, X.device)
            y_m=torch.sparse.mm(trans_M_csr, X.unsqueeze(1)).squeeze(1)
            
            y_out=y_uv+y_m
        else:
            
            x_repeated_for_v = torch.sparse.mm(transVX_in_csr, X.unsqueeze(1)).squeeze(1)
            #print(V_flat.shape,x_repeated_for_v.shape)
            mul_v = V_flat * x_repeated_for_v
    
            mul_vx=torch.sparse.mm(transVX_out_csr, mul_v.unsqueeze(1)).squeeze(1)
            vx_repeated_for_u=torch.sparse.mm(transUX_in_csr, mul_vx.unsqueeze(1)).squeeze(1)
            mul_u=U_flat*vx_repeated_for_u
            mul_ux=torch.sparse.mm(transUX_out_csr, mul_u.unsqueeze(1)).squeeze(1)
            y_uv=torch.sparse.mm(trans_y_csr, mul_ux.unsqueeze(1)).squeeze(1)
            #print(self.trans_M_csr.dtype,X.dtype)
            
            y_m=torch.sparse.mm(trans_M_csr, X.unsqueeze(1)).squeeze(1)
            #print(y_out)
            y_out=y_uv+y_m
        return y_out

        
        






    #def tranfser_xvector(self,X):



        


    def init_UV(self, U_list, V_list):

        if len(U_list) != len(V_list) or len(U_list) == 0:
            raise ValueError("U_list and V_list must have same non-zero length")

        # self.device = device if device is not None else U_list[0].device
        # self.num_batches = len(U_list)

        
        # 安全检查：U 和 V 的秩必须匹配
        # for i, (U, V) in enumerate(zip(U_list, V_list)):
        #     if U.shape[1] != V.shape[0]:
        #         raise ValueError(f"Block {i}: U.shape[1] ({U.shape[1]}) != V.shape[0] ({V.shape[0]})")

        # ====================== 预展平（只做一次） ======================
        U_flat = torch.cat([U.flatten() for U in U_list]).to(self.device)   # shape: (∑ N1_i * R_i,)
        V_flat = torch.cat([V.flatten() for V in V_list]).to(self.device)   # shape: (∑ R_i * N2_i,)

        
        return U_flat,V_flat