import numpy as np
import torch
import scipy.spatial
import os, sys

class Coords_rosetta6d():
    def __init__(self):
        super(Coords_rosetta6d, self).__init__()
        
        
    def get_angles_tensor(self,Cb_Ca, delta_Cb_2D,contact_mask,n_res,n_decoy):
        Cb_Ca /= torch.linalg.norm(Cb_Ca, dim=-1).unsqueeze(dim=-1) # (n_decoy, n_res, 3)
        delta_Cb_2D += 1e-6
        delta_Cb_2D /= torch.linalg.norm(delta_Cb_2D, dim=-1).unsqueeze(dim=-1) # (n_decoy, n_res, n_res, 3)
        Cb_Ca = Cb_Ca.unsqueeze(dim=2).repeat(1,1,n_res,1) # (n_decoy, n_res, n_res, 3)
        Cb_Ca = torch.transpose(Cb_Ca,1,2)
        
        pi = torch.zeros(n_decoy,n_res,n_res) # for innner-product
        for i in range(n_decoy):
            a = Cb_Ca[i,contact_mask[i]]
            b = delta_Cb_2D[i,contact_mask[i]]
            inprod = torch.sum(a * b, dim=-1)
            
            tmp = torch.zeros(n_res,n_res)
            tmp[contact_mask[i]] = torch.acos(inprod)
            pi[i] = tmp

        return pi


    # For edge features (pairwise calculation)
    def get_dihedrals_tensor(self,p1,q1,r1,contact_mask,n_res,n_decoy):
        inprod = torch.zeros(n_decoy,n_res,n_res)
        for i in range(n_decoy):
            p = p1[i,contact_mask[i]]
            r = r1[i,contact_mask[i]] 
            # p, q : both [#contact,xyz]
            q = q1[i,contact_mask[i]] # [#contact,xyz]
            
            v1 = torch.sum(p*q, dim=-1).unsqueeze(dim=-1)
            # [#contact] -> (After unsqueeze) -> [#contact,1]
            v2 = v1*q # [#contact,xyz]
            v3 = p-v2 # [#contact,xyz]
            
            # both [#contact,xyz]
            v = p - torch.sum(p * q, dim=-1).unsqueeze(dim=-1) * q
            w = r - torch.sum(q * r, dim=-1).unsqueeze(dim=-1) * q
            
            x = torch.sum(v * w, dim=-1)
            
            # cross:[#contact,xyz], r:[#contact,xyz]
            y = torch.sum((torch.linalg.cross(q,v)*r),dim=-1)  # [#contact,xyz]
            
            tmp = torch.zeros(n_res,n_res)
            tmp[contact_mask[i]] = torch.atan2(y,x)
            inprod[i] = tmp
        
        return inprod # inprod [n_decoy, n_res*n_res]
    
    
    def get_dihedrals_node(self,p,q,r,n_res,n_decoy):
        v1 = torch.sum(p*q, dim=-1).unsqueeze(dim=-1) #[n_decoy, n_res] -> (After unsqueeze) -> [n_decoy, n_res, 1]
        v2 = v1*q #[n_decoy,n_res,3]
        v3 = p-v2 #
        
        v = p - torch.sum(p*q, dim=-1).unsqueeze(dim=-1)*q
        w = r - torch.sum(q*r, dim=-1).unsqueeze(dim=-1)*q
        # both [n_decoy, n_res, 3]
        
        x = torch.sum(v*w, dim=-1)
        y = torch.sum((torch.linalg.cross(q,v)*r),dim=-1)
        # both [n_decoy, n_res]
        
        angle = torch.atan2(y,x)        
        
        return angle # [n_decoy, n_res]
        
        
    def get_angles(self,a,b,c):
        # get angles from three points a, b, c
        v = a - b
        v /= torch.linalg.norm(v)
        w = c - b
        w /= torch.linalg.norm(w)
        x = torch.inner(v,w)
        
        return torch.acos(x)


    def get_dihedrals(self,a,b,c,d):
        # get dihedral angles from four points a, b, c, d
        p = a - b
        q = c - b
        q /= torch.linalg.norm(q)
        r = d - c
        
        v = p - torch.inner(p,q)*q
        w = r - torch.inner(q,r)*q
        
        x = torch.inner(v,w)
        y = torch.inner(torch.linalg.cross(q,v),r)
        
        return torch.atan2(y,x)


    def get_coords6d(self,coords,is_gly,dmax):
        # coords shape [n_decoy, n_res, 4, 5] (x,y,z,aatype,atomtype)
        coords = coords[:,:,:,:3]
        n_res = coords.shape[1]
        n_decoy = coords.shape[0]
        
        N = coords[:,:,0]
        Ca = coords[:,:,1]
        C = coords[:,:,2]
        O = coords[:,:,3]
        Cb = coords[:,:,4]
        
        # Use virtual Cb for GLY from trRosetta2 -> to entire sequence ? Cb 제외 그냥 O로 대체
        b = Ca - N
        c = C - Ca
        a = torch.linalg.cross(b,c, dim=-1) # [n_decoy, n_res, 3] 
        tmp = -0.58273431*a[:,is_gly] + 0.56802827*b[:,is_gly] - 0.54067466*c[:,is_gly] + Ca[:,is_gly]
        Cb[:,is_gly]=torch.FloatTensor(tmp)
        
        # Check contact residues (distance < dmax)
        contact_mask = torch.zeros(n_decoy,n_res,n_res)
        Cb = Cb.unsqueeze(dim=2) # [n_decoy, n_res, 1, 3]
        Cb_2 = Cb.repeat(1,1,n_res,1) # [n_decoy, n_res, n_res, 3]
        Cb_1 = torch.transpose(Cb_2,1,2)
        dist_Cb = torch.linalg.norm(Cb_1 - Cb_2, dim=-1) # [n_decoy, n_res, n_res]
        Cb = Cb.squeeze(dim=2)
        contact_mask[dist_Cb < dmax] = 1
        contact_mask[dist_Cb == 0] = 0
        contact_mask = contact_mask.bool()
    
        
        # contact mask for Ca-Ca distance dmax(15?)
        '''contact_mask = torch.zeros(n_decoy,n_res,n_res)
        Ca = Ca.unsqueeze(dim=2) # [n_decoy, n_res, 1, 3]
        Ca_2 = Ca.repeat(1,1,n_res,1)
        Ca_1 = torch.transpose(Ca_2,1,2)
        dist_Ca = torch.linalg.norm(Ca_1-Ca_2,dim=-1)
        Ca=Ca.squeeze(dim=2)
        contact_mask[dist_Ca < dmax] = 1
        contact_mask[dist_Ca==0] = 0
        contact_mask = contact_mask.bool()
        print('contact mask ',contact_mask.shape, contact_mask)
        sys.exit()'''
        
        # Indicies of residues in contact
        y = torch.arange(n_res).unsqueeze(1).repeat(1,n_res)
        x = torch.transpose(y,0,1)
        idx = torch.stack([x,y],dim=0)
        idx_contact = torch.zeros(n_decoy,2,n_res*n_res)
        for i in range(n_decoy):
            idx1 = idx[0,contact_mask[i]]
            idx2 = idx[1,contact_mask[i]]
            idx_contact[i,0,:idx1.size(0)]=idx1
            idx_contact[i,1,:idx2.size(0)]=idx2
        
        
        # Make coordinates 1D to 2D
        Cb_Ca = Ca - Cb # (n_decoy,n_res,3)
        delta_Cb_2D = Cb_2 - Cb_1 # (n_decoy,n_res,n_res,3)
        N_Ca = Ca - N # (n_decoy,n_res,3)
        Ca_C = C - Ca # (n_decoy,n_res,3)
            
        # Cb-Cb distance matrix
        dist6d = torch.full_like(dist_Cb, fill_value = 999.9)
        dist6d[contact_mask] = dist_Cb[contact_mask] # [n_decoy, n_res, n_res]
        
        # Matrix of polar coord phi (Ca-Cb-Cb)
        phi6d = self.get_angles_tensor(Cb_Ca, delta_Cb_2D,contact_mask,n_res,n_decoy) # (n_decoy, n_res**2)

        # Matrix of Ca-Cb-Cb-Ca dihedrals omega
        omega_2 = delta_Cb_2D + 1e-6
        omega_2 /= torch.linalg.norm(omega_2, dim=-1).unsqueeze(dim=-1)
        omega_1 = Cb_Ca.unsqueeze(dim=2).repeat(1,1,n_res,1)
        omega_3 = torch.transpose(omega_1,1,2)
        omega6d = self.get_dihedrals_tensor(omega_1,omega_2,omega_3,contact_mask,n_res,n_decoy)
        
        # Matrix of polar coord theta (N1-Ca1-Cb1-Cb2)
        theta_1 = (-N_Ca).unsqueeze(dim=2).repeat(1,1,n_res,1)
        theta_2 = delta_Cb_2D + 1e-6
        theta_2 /= torch.linalg.norm(theta_2, dim=-1).unsqueeze(dim=-1)
        theta_3 = Cb_Ca.unsqueeze(dim=1).repeat(1,n_res,1,1)
        theta6d = self.get_dihedrals_tensor(theta_1,theta_2,theta_3,contact_mask,n_res,n_decoy) #[n_decoy, n_res, n_res]
        
        # Matrix of polar coord phi of residue feature (C0-N-Ca-C)
        C0 = torch.roll(C,shifts=1,dims=1)
        C0[:,-1,0]=0; C0[:,-1,1]=0; C0[:,-1,2]=0
        phi_1 = C0 - N #[n_decoy, n_res, 3]
        phi_2 = N_Ca + 1e-6 #[n_decoy, n_res, 3]
        phi_2 /= torch.linalg.norm(phi_2, dim=-1).unsqueeze(dim=-1)
        phi_3 = Ca_C
        phi_res = self.get_dihedrals_node(phi_1,phi_2,phi_3,n_res,n_decoy) #[n_decoy, n_res]
        
        # Matrix of polar coord psi of residue feature (N-Ca-C-N2)
        N2 = torch.roll(N,shifts=-1,dims=1)
        N2[:,0,0]=0; N2[:,0,1]=0; N2[:,0,2]=0
        psi_1 = -N_Ca #[n_decoy, n_res, 3]
        psi_2 = Ca_C + 1e-6
        psi_2 /= torch.linalg.norm(psi_2, dim=-1).unsqueeze(dim=-1)
        psi_3 = N2 - C
        psi_res = self.get_dihedrals_node(psi_1,psi_2,psi_3,n_res,n_decoy) #[n_decoy, n_res]
        
        return dist6d, phi6d, omega6d, theta6d, phi_res, psi_res