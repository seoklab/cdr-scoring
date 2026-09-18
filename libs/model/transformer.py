import logging
from typing import Optional, Literal, Dict

import torch
import torch.nn as nn
from dgl import DGLGraph
import dgl
from torch import Tensor

from model.basis import get_basis, update_basis_with_fused
from model.layers.attention import AttentionBlockSE3
from model.layers.convolution import ConvSE3, ConvSE3FuseLevel
from model.layers.norm import NormSE3
from model.layers.pooling import GPooling
from runtime.utils import str2bool
from model.fiber import Fiber
from config import config as se3_config

from runtime.constants import *
import traceback

DROP_OUT_RATE=0.2

class Sequential(nn.Sequential):
    """ Sequential module with arbitrary forward args and kwargs. Used to pass graph, basis and edge features. """

    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input
class GDropOut(nn.Dropout):
    def forward(self, h: dict,*__, **_):
        h_d = {}
        for k, v in h.items():
            h_d[k] = super().forward(v)
        return h_d


def get_populated_edge_features(relative_pos: Tensor, edge_features: Optional[Dict[str, Tensor]] = None):
    """ Add relative positions to existing edge features """
    edge_features = edge_features.copy() if edge_features else {}
    r = relative_pos.norm(dim=-1, keepdim=True)
    if '0' in edge_features:
        edge_features['0'] = torch.cat([edge_features['0'], r[..., None]], dim=1)
    else:
        edge_features['0'] = r[..., None]

    return edge_features

class SE3Transformer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 fiber_in: Fiber,
                 fiber_hidden: Fiber,
                 fiber_out: Fiber,
                 num_heads: int,
                 channels_div: int,
                 fiber_edge: Fiber = Fiber({}),
                 return_type: Optional[int] = None,
                 pooling: Optional[Literal['avg', 'max']] = None,
                 norm: bool = True,
                 use_layer_norm: bool = True,
                 tensor_cores: bool = False,
                 low_memory: bool = False,
                 mode_monomer: bool =False,
                 use_last_conv:bool=True,
                 use_all_atom:bool=False,
                 **kwargs):
        """
        :param num_layers:          Number of attention layers
        :param fiber_in:            Input fiber description
        :param fiber_hidden:        Hidden fiber description
        :param fiber_out:           Output fiber description
        :param fiber_edge:          Input edge fiber description
        :param num_heads:           Number of attention heads
        :param channels_div:        Channels division before feeding to attention layer
        :param return_type:         Return only features of this type
        :param pooling:             'avg' or 'max' graph pooling before MLP layers
        :param norm:                Apply a normalization layer after each attention block
        :param use_layer_norm:      Apply layer normalization between MLP layers
        :param tensor_cores:        True if using Tensor Cores (affects the use of fully fused convs, and padded bases)
        :param low_memory:          If True, will use slower ops that use less memory
        """
        super().__init__()
        self.num_layers = num_layers
        self.fiber_edge = fiber_edge
        self.num_heads = num_heads
        self.channels_div = channels_div
        self.return_type = return_type
        self.pooling = pooling
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees)
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory
        self.use_last_conv=use_last_conv
        self.use_all_atom=use_all_atom
        if low_memory:
            self.fuse_level = ConvSE3FuseLevel.NONE
        else:
            # Fully fused convolutions when using Tensor Cores (and not low memory mode)
            self.fuse_level = ConvSE3FuseLevel.FULL if tensor_cores else ConvSE3FuseLevel.PARTIAL
        ##MUST
        graph_modules = []
        for i in range(num_layers):
            graph_modules.append(AttentionBlockSE3(fiber_in=fiber_in,
                                                   fiber_out=fiber_hidden,
                                                   fiber_edge=fiber_edge,
                                                   num_heads=num_heads,
                                                   channels_div=channels_div,
                                                   use_layer_norm=use_layer_norm,
                                                   max_degree=self.max_degree,
                                                   fuse_level=self.fuse_level,
                                                   low_memory=low_memory))
            if norm:
                graph_modules.append(NormSE3(fiber_hidden))
                graph_modules.append(GDropOut(p=DROP_OUT_RATE))
            fiber_in = fiber_hidden
        graph_modules.append(ConvSE3(fiber_in=fiber_in,
                                     fiber_out=fiber_out,
                                     fiber_edge=fiber_edge,
                                     self_interaction=True,
                                     use_layer_norm=use_layer_norm,
                                     max_degree=self.max_degree,
                                     fuse_level=self.fuse_level,
                                     low_memory=low_memory))
        self.graph_modules = Sequential(*graph_modules)
        if pooling is not None:
            assert return_type is not None, 'return_type must be specified when pooling'
            self.pooling_module = GPooling(pool=pooling, feat_type=return_type)

    def forward(self, graph: DGLGraph, node_feats: Dict[str, Tensor],
                edge_feats: Optional[Dict[str, Tensor]] = None,
                basis: Optional[Dict[str, Tensor]] = None):
        # Compute bases in case they weren't precomputed as part of the data loading
        if self.use_all_atom:
            basis = basis or get_basis(graph.edata['rel_ca_pos'], max_degree=self.max_degree, compute_gradients=False,
                                   use_pad_trick=self.tensor_cores and not self.low_memory,
                                   amp=torch.is_autocast_enabled())
        else:
            basis = basis or get_basis(graph.edata['rel_pos'], max_degree=self.max_degree, compute_gradients=False,
                                   use_pad_trick=self.tensor_cores and not self.low_memory,
                                   amp=torch.is_autocast_enabled())

        # Add fused bases (per output degree, per input degree, and fully fused) to the dict
        basis = update_basis_with_fused(basis, self.max_degree, use_pad_trick=self.tensor_cores and not self.low_memory, fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)
        if self.use_all_atom:
            edge_feats = get_populated_edge_features(graph.edata['rel_ca_pos'], edge_feats)
        else:
            edge_feats = get_populated_edge_features(graph.edata['rel_pos'], edge_feats)
        node_feats = self.graph_modules(node_feats, edge_feats, graph=graph, basis=basis)

        self.return_type=None
        if self.pooling is not None:
            return self.pooling_module(node_feats, graph=graph)
        if self.return_type is not None:
            return node_feats[str(self.return_type)]
        return node_feats

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--num_layers', type=int, default=4,
                            help='Number of stacked Transformer layers')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of heads in self-attention')
        parser.add_argument('--channels_div', type=int, default=2,
                            help='Channels division before feeding to attention layer')
        parser.add_argument('--pooling', type=str, default=None, const=None, nargs='?', choices=['max', 'avg'],
                            help='Type of graph pooling')
        parser.add_argument('--norm', type=str2bool, nargs='?', const=True, default=True,
                            help='Apply a normalization layer after each attention block')
        parser.add_argument('--use_layer_norm', type=str2bool, nargs='?', const=True, default=True,
                            help='Apply layer normalization between MLP layers')
        parser.add_argument('--low_memory', type=str2bool, nargs='?', const=True, default=False,
                            help='If true, will use fused ops that are slower but that use less memory '
                                 '(expect 25 percent less memory). '
                                 'Only has an effect if AMP is enabled on Volta GPUs, or if running on Ampere GPUs')

        return parser
def _get_relative_pos(graph: DGLGraph) -> Tensor:
    x = graph.ndata['pos']
    src, dst = graph.edges()
    rel_pos = x[dst] - x[src]
    return rel_pos



class Sujin_with_SE3(nn.Module):
    def __init__(self,
                 fiber_in: Fiber,
                 fiber_out: Fiber,
                 fiber_edge: Fiber,
                 use_nodewise_score: bool,
                 num_degrees: int,
                 num_channels: int,
                 num_layers=4,
                 **kwargs):
        super(Sujin_with_SE3,self).__init__()

        '''
        From sujin's original code
        '''
        # control variable with defined fiber
        initial_node_dim = 8+4 
        initial_edge_dim = 25+2+1+6
        #
        embed_dim=8
        self.embed_layer = torch.nn.Embedding(20,embed_dim)
        self.bond_embed_layer = torch.nn.Linear(2,1,bias=False)
        #hidden_dim is defined by fiber
        self.embedding_node = nn.Linear(initial_node_dim, fiber_in[0], bias=False)
        self.embedding_edge = nn.Linear(initial_edge_dim, fiber_edge[0], bias=False)
        # self.readout='max'
        # self.readout = 'sum' 
        self.readout='mean'
        self.mp_layers = torch.nn.ModuleList()
        #
        self.linear_out_1 = nn.Linear(fiber_out[0], 1, bias=False)  # intrinsic loop-quality head
        self.ord_head = nn.Linear(fiber_out[0], 3, bias=True)       # H3 ordinal head
        # v2 multi-head (shared pooled embedding):
        #   interface_head : interface-compatibility scalar (defined now; inactive in
        #                    v2 pretrain — label = GP loop lDDT / holo DockQ, wired later)
        #   final_head     : Phase-C antibody ranking score (trained in DPO finetune)
        self.interface_head = nn.Linear(fiber_out[0], 1, bias=False)
        self.final_head = nn.Linear(fiber_out[0], 1, bias=False)
        #
        self.num_layers = num_layers
        self.act_fn=nn.ReLU()
        self.use_nodewise_score=use_nodewise_score
        #
        '''
        From HU 
        '''
        self.norm_node=nn.LayerNorm(int(fiber_in[0]))
        self.norm_edge=nn.LayerNorm(int(fiber_edge[0]))
        ###
        print('### transformer fiber in ',fiber_in)
        print('### transformer fiber edge ',fiber_edge)
        self.transformer = SE3Transformer(
            fiber_in=fiber_in,#fiber_in={0:24,1:4}
            fiber_hidden=Fiber.create(num_degrees, num_channels), #Fiber.create(2,64) => {0:64, 1:64}
            fiber_out=fiber_out,
            # fiber_edge=Fiber({0:fiber_edge[0]}),
            fiber_edge=fiber_edge,
            return_type=0,#
            num_layers=self.num_layers,
            **kwargs
        )

    def forward(self, batched_graph,basis=None):
        '''
        Input Embedding from Sujin
        '''
        #node embedding
        #graph information
        #graph.ndata['pos'] : CA position coordinate
        #graph.ndata['l1'] : Ca->N,Ca->O,Ca->C,Ca->Cb
        #dummy
        try:
            # batched_graph.ndata['pos']=torch.zeros(batched_graph.ndata['bb_torsion'].shape[0],3).to(device=batched_graph.ndata['bb_torsion'].device)
            batched_graph.ndata['pos'].to(device=batched_graph.ndata['bb_torsion'].device)
            # type-0: aa_feature[8], bb_torsion[2+2]
            aatype = batched_graph.ndata['h'].float()
            aa_feature = self.embed_layer(aatype.int()) # (3961,8)
            bb_torsion = batched_graph.ndata['bb_torsion'].float()
            node = torch.cat([aa_feature,torch.sin(bb_torsion),torch.cos(bb_torsion)],dim=1)
            del aatype, aa_feature, bb_torsion
            node = self.embedding_node(node)
            #edge embedding type-0: distance[25], bond_character[2], rel_index[1], pair_torsion[3+3]
            bond_character = batched_graph.edata['e_ij'][:,-2:].float()
            bond_character = self.bond_embed_layer(bond_character).float()
            distance = batched_graph.edata['e_ij'][:,:25].float()
            distance = 1/(1+distance**2)
            rel_index = batched_graph.edata['rel_index'].float()
            pair_torsion = batched_graph.edata['pair_torsion'].float()
            mask = torch.isnan(pair_torsion)
            pair_torsion = pair_torsion.masked_fill_(mask,value=1e-6)
            edge = torch.cat([distance,bond_character,rel_index,torch.sin(pair_torsion),torch.cos(pair_torsion)],dim=1)
            del distance, bond_character, rel_index, pair_torsion, mask
            edge = self.embedding_edge(edge)
            '''
            SE3
            '''
            #calculate relative position along with edge/ it is required for basis calculation for SE3 transformer
            batched_graph.edata['rel_pos']=_get_relative_pos(batched_graph)
            #make input feature float64 //if you already parse your input feature as float64 it may not be requried
            # node_feats={'0':node}
            node_feats={'0':node,'1':batched_graph.ndata['l1'].float()}
            edge_feats={'0':edge, '1':batched_graph.edata['rel_pos'].float()}
            #normalization &  modify shape of tensor to l=0 feature form// l=0 1D rotation equivariant : scalar // l=1
            #  
            node_feats['0']=self.norm_node(self.act_fn(node_feats['0'])).unsqueeze(-1)#l=0 // N,C_m,1 
            node_feats['1']=node_feats['1'] # l=1 // N,C_m,3  || l=2 // N,C_m,5 2l+1
            edge_feats['0']=self.norm_edge(self.act_fn(edge_feats['0'])).unsqueeze(-1)
            edge_feats['1']=edge_feats['1'].unsqueeze(-2)
            #run se3 transformer
            feats = self.transformer(batched_graph, node_feats, edge_feats, basis)
            #####
            batched_graph.ndata['out_l0']=feats['0'].squeeze(-1)
            out_dic = {}
            out = dgl.readout_nodes(batched_graph, 'out_l0', op=self.readout)
            out_dic['out'] = self.linear_out_1(out).squeeze(-1)     # intrinsic (== pretrain ranking score)
            out_dic['ord_logits'] = self.ord_head(out)
            # v2 heads (shared pooled embedding `out`). `out`/intrinsic drive the
            # pretrain SML; interface/final are emitted for downstream phases.
            out_dic['intrinsic'] = out_dic['out']
            # pooled backbone embedding, before any head. Additive only -- used by
            # analyze/probe_backbone_fnat.py to test what the trunk still encodes.
            out_dic['embed'] = out
            out_dic['interface'] = self.interface_head(out).squeeze(-1)
            out_dic['final'] = self.final_head(out).squeeze(-1)
            if self.use_nodewise_score:
                node = batched_graph.ndata['out_l0']
                score = self.linear_out_1(node).squeeze(-1)
                num_nodes_per_graph = batched_graph.batch_num_nodes().tolist()
                nodewise_score_split = torch.split(score, num_nodes_per_graph)
                ulr_feature = batched_graph.ndata['ulr']
                current_idx = 0
                nodewise_score_filtered = []
                for i, graph_nodes in enumerate(nodewise_score_split):
                    graph_ulr = ulr_feature[current_idx:current_idx + num_nodes_per_graph[i]]
                    filtered_nodes = graph_nodes[graph_ulr == 1]
                    nodewise_score_filtered.append(filtered_nodes)
                    current_idx += num_nodes_per_graph[i]
                out_dic['nodewise_score'] = nodewise_score_filtered
        except Exception as e:
            traceback.print_exc()
            raise
        #out_dic['out_cee']=out_cee
        #out_dic['out_sml']=out_sml
        return out_dic

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model architecture")
        SE3Transformer.add_argparse_args(parser)
        parser.add_argument('--num_degrees',
                            help='Number of degrees to use. Hidden features will have types [0, ..., num_degrees - 1]',
                            type=int, default=2)
        parser.add_argument('--num_channels', help='Number of channels for the hidden features', type=int, default=32)
        return parent_parser



def generate_all_atom_edge_features(batched_graph):
    """
    Generate all-atom edge features (14x14) for each edge in the graph, based 
    on atom positions.
    
    :param batched_graph: DGLGraph containing the node and edge features.
    :return: None (modifies the graph in-place to add the all-atom edge feature).
    """
    src, dst = batched_graph.edges()
    all_atom_edge_vectors = []
    
    for s, d in zip(src, dst):
        pos_s = batched_graph.ndata['all_atom_rel_pos'][s]
        pos_d = batched_graph.ndata['all_atom_rel_pos'][d]

        pairwise_vectors = pos_s[:, None, :] - pos_d[None, :, :]
        all_atom_edge_vectors.append(pairwise_vectors)
        
    all_atom_edge_vectors = torch.stack(all_atom_edge_vectors, dim=0)
    batched_graph.edata['all_atom_edge_l1'] = all_atom_edge_vectors

    return batched_graph



class Sujin_with_SE3_allatom(nn.Module):
    def __init__(self,
                 fiber_in: Fiber,
                 fiber_out: Fiber,
                 fiber_edge: Fiber,
                 num_degrees: int,
                 num_channels: int,
                 num_layers=4,
                 **kwargs):
        super(Sujin_with_SE3_allatom,self).__init__()
        '''
        From sujin's original code
        '''
        # control variable with defined fiber
        initial_node_dim = 8+4+14 
        initial_edge_dim = 1+1+6
        #
        aa_embed_dim=8
        atom_embed_dim=4
        self.aa_embed_layer = torch.nn.Embedding(20,aa_embed_dim)
        self.atom_embed_layer = torch.nn.Embedding(MAX_NUM_ATOM,atom_embed_dim)
        self.bond_embed_layer = torch.nn.Linear(2,1,bias=False)
        self.index_embed_layer = torch.nn.Linear(2,1,bias=False)
        self.all_atom_edge_layer = torch.nn.Linear(14*14, fiber_edge[0], bias=False)
        #hidden_dim is defined by fiber
        self.embedding_node = nn.Linear(initial_node_dim, fiber_in[0], bias=False)
        self.embedding_edge = nn.Linear(initial_edge_dim, fiber_edge[0], bias=False)
        self.readout='mean'
        self.mp_layers = torch.nn.ModuleList()
        #
        self.linear_out_1 = nn.Linear(fiber_out[0], 1, bias=False)  # intrinsic loop-quality head
        self.ord_head = nn.Linear(fiber_out[0], 3, bias=True)       # H3 ordinal head
        # v2 multi-head (shared pooled embedding); see Sujin_with_SE3 for semantics.
        self.interface_head = nn.Linear(fiber_out[0], 1, bias=False)
        self.final_head = nn.Linear(fiber_out[0], 1, bias=False)
        #
        self.num_layers = num_layers
        self.act_fn=nn.ReLU()
        #
        
        '''
        From HU 
        '''
        self.norm_node=nn.LayerNorm(int(fiber_in[0]))
        self.norm_edge=nn.LayerNorm(int(fiber_edge[0]))
        ###
        print('### transformer fiber in ',fiber_in)
        print('### transformer fiber edge ',fiber_edge)
        self.transformer = SE3Transformer(
            fiber_in=fiber_in,#fiber_in={0:24,1:4}
            fiber_hidden=Fiber.create(num_degrees, num_channels), #Fiber.create(2,64) => {0:64, 1:64}
            fiber_out=fiber_out,
            # fiber_edge=Fiber({0:fiber_edge[0]}),
            fiber_edge=fiber_edge,
            return_type=0,#
            num_layers=self.num_layers,
            use_all_atom=True,
            **kwargs
        )

    

    def forward(self, batched_graph, basis=None):
        '''
        Input Embedding from Sujin
        '''
        #node embedding
        #graph.ndata['pos'] : CA position coordinate
        #graph.ndata['l1'] : Ca->N,Ca->O,Ca->C,Ca->Cb

        batched_graph = generate_all_atom_edge_features(batched_graph)

        
        # type-0: aa_feature[8], bb_torsion[2+2], all_atom_type[4] 
        aatype = batched_graph.ndata['h'].float()
        aa_feature = self.aa_embed_layer(aatype.int()) # (3961,8)
        bb_torsion = batched_graph.ndata['bb_torsion'].float()
        all_atom_type = batched_graph.ndata['info_aa_atom'][:,:,0].float() #14
        node = torch.cat([aa_feature,torch.sin(bb_torsion),torch.cos(bb_torsion),all_atom_type],dim=1)
        node = self.embedding_node(node)
        
        #edge embedding type-0: distance[25], bond_character[2], rel_index[1], pair_torsion[3+3]
        bond_character = batched_graph.edata['e_ij'][:,-2:].float()
        bond_character = self.bond_embed_layer(bond_character).float()
        rel_index = batched_graph.edata['rel_index'].float()
        rel_index = self.index_embed_layer(rel_index).float()
        pair_torsion = batched_graph.edata['pair_torsion'].float()
        mask = torch.isnan(pair_torsion)
        pair_torsion = pair_torsion.masked_fill_(mask,value=1e-6).float()
        edge = torch.cat([bond_character,rel_index,torch.sin(pair_torsion),torch.cos(pair_torsion)],dim=1)
        edge = self.embedding_edge(edge)

        all_atom_edge_feat = batched_graph.edata['all_atom_edge'].float()
        all_atom_edge_feat = all_atom_edge_feat.view(all_atom_edge_feat.shape[0],-1)
        all_atom_edge_feat = self.all_atom_edge_layer(all_atom_edge_feat)

        edge = torch.cat([edge,all_atom_edge_feat],dim=1)
        '''
        SE3
        '''
        #calculate relative position along with edge/ it is required for basis calculation for SE3 transformer
        # batched_graph.edata['rel_pos']=_get_relative_pos(batched_graph)
        #make input feature float64 //if you already parse your input feature as float64 it may not be requried
        # node_feats={'0':node}
        node_feats={'0':node,'1':batched_graph.ndata['all_atom_rel_pos'].float()}
        edge_feats={'0':edge, '1':batched_graph.edata['rel_ca_pos'].float()}
        #normalization &  modify shape of tensor to l=0 feature form// l=0 1D rotation equivariant : scalar // l=1
        #  
        node_feats['0']=self.norm_node(self.act_fn(node_feats['0'])).unsqueeze(-1)#l=0 // N,C_m,1
        node_feats['1']=node_feats['1'] # l=1 // N,C_m,3  || l=2 // N,C_m,5 2l+1
        edge_feats['0']=self.norm_edge(self.act_fn(edge_feats['0'])).unsqueeze(-1)
        edge_feats['1']=edge_feats['1'].unsqueeze(-2)
        #run se3 transformer
        feats = self.transformer(batched_graph, node_feats, edge_feats, basis)
        #####
        batched_graph.ndata['out_l0']=feats['0'].squeeze(-1) 
        '''
        readout
        '''
        out = dgl.readout_nodes(batched_graph, 'out_l0', op=self.readout)
        #####
        out_dic={}
        out_dic['out']=self.linear_out_1(out).squeeze(-1)
        out_dic['ord_logits'] = self.ord_head(out)
        out_dic['intrinsic'] = out_dic['out']
        out_dic['embed'] = out          # pooled backbone embedding, before any head
        out_dic['interface'] = self.interface_head(out).squeeze(-1)
        out_dic['final'] = self.final_head(out).squeeze(-1)
        #out_dic['out_cee']=out_cee
        #out_dic['out_sml']=out_sml
        return out_dic

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Model architecture")
        SE3Transformer.add_argparse_args(parser)
        parser.add_argument('--num_degrees',
                            help='Number of degrees to use. Hidden features will have types [0, ..., num_degrees - 1]',
                            type=int, default=2)
        parser.add_argument('--num_channels', help='Number of channels for the hidden features', type=int, default=32)
        return parent_parser

