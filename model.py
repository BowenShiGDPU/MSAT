import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MSATEdgeEncoder(nn.Module):
 
    
    def __init__(self, edge_dim=6, num_heads=8, hidden_dim=64):
        super().__init__()
        
        
        self.mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_heads)
        )
        
        
        self.shortcut = nn.Linear(edge_dim, num_heads)
        
        
        self.gate_param = nn.Parameter(torch.tensor(-0.5))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.shortcut.weight)
        nn.init.zeros_(self.shortcut.bias)
        
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
    
    def forward(self, edge_attr):
        mlp_output = self.mlp(edge_attr)
        shortcut_output = self.shortcut(edge_attr)
        
        
        gate = torch.sigmoid(self.gate_param)
        return gate * mlp_output + (1 - gate) * shortcut_output


class MSATOutputTransform(nn.Module):
   
    
    def __init__(self, out_channels):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),      
            nn.ReLU(),
            nn.Linear(out_channels * 2, out_channels * 3),  
            nn.ReLU(),
            nn.Linear(out_channels * 3, out_channels)       
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        
        for i, module in enumerate(self.mlp):
            if isinstance(module, nn.Linear):
                if i == 0:
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    nn.init.zeros_(module.bias)
                elif i == 2:
                    nn.init.xavier_uniform_(module.weight, gain=0.8)
                    nn.init.zeros_(module.bias)
                elif i == 4:
                    nn.init.xavier_uniform_(module.weight, gain=0.6)
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.mlp(x)


class LateFusionPredictor(nn.Module):
   
    
    def __init__(self, emb_dim=72, dropout=0.18):
        super().__init__()
        
        
        self.fusion_weights = nn.Parameter(torch.ones(3))
        
        
        self.mlp_head = nn.Sequential(
            nn.Linear(emb_dim * 2 + 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1)
        )
        
       
        self.bilinear_head = nn.Bilinear(emb_dim, emb_dim, 1)
        
       
        self.distmult_head = nn.Linear(emb_dim, 1)
    
    def forward(self, herb_emb, adr_emb, herb_degree, adr_degree):
       
        mlp_input = torch.cat([herb_emb, adr_emb, herb_degree, adr_degree], dim=-1)
        mlp_out = self.mlp_head(mlp_input)
        
       
        bilinear_out = self.bilinear_head(herb_emb, adr_emb)
        
       
        hadamard = herb_emb * adr_emb
        distmult_out = self.distmult_head(hadamard)
        
       
        fusion_w = F.softmax(self.fusion_weights, dim=0)
        final_logits = fusion_w[0] * mlp_out + fusion_w[1] * bilinear_out + fusion_w[2] * distmult_out
        
        return final_logits.squeeze(-1)


class MultiTypeGraphAttention(nn.Module):

    
    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 8,
                 dropout: float = 0.18, edge_dim: int = 6):
        super().__init__()
        
        assert out_channels % num_heads == 0
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        
        
        self.W_q = nn.Linear(in_channels, out_channels)
        self.W_k = nn.Linear(in_channels, out_channels)
        self.W_v = nn.Linear(in_channels, out_channels)
        
       
        self.edge_encoder = MSATEdgeEncoder(edge_dim, num_heads, hidden_dim=64)
        
        
        self.output_mlp = MSATOutputTransform(out_channels)
        
        
        self.pos_encoding = nn.Parameter(torch.randn(1, out_channels))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.zeros_(self.W_k.bias)
        nn.init.zeros_(self.W_v.bias)
        
        self.edge_encoder.reset_parameters()
        self.output_mlp.reset_parameters()
    
    def forward(self, x_src, x_dst, edge_index, edge_attr=None):
        num_dst_nodes = x_dst.size(0)
        
       
        Q_dst = self.W_q(x_dst).view(-1, self.num_heads, self.head_dim)
        K_src = self.W_k(x_src).view(-1, self.num_heads, self.head_dim)
        V_src = self.W_v(x_src).view(-1, self.num_heads, self.head_dim)
        
        src, dst = edge_index[0], edge_index[1]
        
        Q_edges = Q_dst[dst]
        K_edges = K_src[src]
        V_edges = V_src[src]
        
      
        attention_scores = torch.sum(Q_edges * K_edges, dim=-1) / math.sqrt(self.head_dim)
        
     
        if edge_attr is not None:
            edge_bias = self.edge_encoder(edge_attr)
            attention_scores = attention_scores + edge_bias
        
       
        attention_weights = F.softmax(attention_scores, dim=0)
        attention_weights = self.dropout(attention_weights)
        
      
        out = torch.zeros(num_dst_nodes, self.out_channels, device=x_dst.device, dtype=x_dst.dtype)
        messages = (V_edges * attention_weights.unsqueeze(-1)).reshape(-1, self.out_channels)
        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.out_channels), messages)
        
        out = out.view(-1, self.num_heads, self.head_dim)
        out = out.view(-1, self.out_channels)
        
        
        out = self.output_mlp(out)
        
       
        if self.in_channels == self.out_channels:
            out = out + x_dst
        out = self.layer_norm(out + self.pos_encoding)
        
        return out


class MultiRelationAttentionLayer(nn.Module):
    
    
    def __init__(self, node_types, edge_types, in_channels, out_channels,
                 num_heads=8, dropout=0.18, edge_attr_dim=6):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.out_channels = out_channels
        
        self.convs = nn.ModuleDict()
        for edge_type in edge_types:
            src_type, rel_type, dst_type = edge_type
            conv_key = f"{src_type}_{rel_type}_{dst_type}"
            
            self.convs[conv_key] = MultiTypeGraphAttention(
                in_channels[src_type],
                out_channels,
                num_heads,
                dropout,
                edge_attr_dim
            )
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        out_dict = {ntype: torch.zeros(
            x_dict[ntype].size(0), self.out_channels,
            device=x_dict[ntype].device, dtype=x_dict[ntype].dtype
        ) for ntype in self.node_types}
        
        for edge_type in self.edge_types:
            src_type, rel_type, dst_type = edge_type
            conv_key = f"{src_type}_{rel_type}_{dst_type}"
            
            if edge_type not in edge_index_dict or edge_index_dict[edge_type].size(1) == 0:
                continue
            
            out_dict[dst_type] = out_dict[dst_type] + self.convs[conv_key](
                x_dict[src_type], x_dict[dst_type],
                edge_index_dict[edge_type], edge_attr_dict.get(edge_type)
            )
        
        return out_dict


class MSATTCMFSFinal(nn.Module):
 
    
    def __init__(self, node_types, edge_types, in_channels_dict,
                 hidden_channels=576, out_channels=72, num_layers=3,
                 num_heads=8, dropout=0.18, edge_attr_dim=6,
                 node_degrees_dict=None):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.num_layers = num_layers
        self.node_degrees_dict = node_degrees_dict or {}
        
       
        self.eps = nn.ParameterDict({
            ntype: nn.Parameter(torch.zeros(1))
            for ntype in node_types
        })
        
       
        self.degree_scale_weight = nn.Parameter(torch.ones(1))
        
     
        self.input_layers = nn.ModuleDict()
        for node_type in node_types:
            self.input_layers[node_type] = nn.Sequential(
                nn.Linear(in_channels_dict[node_type], hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
       
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            in_channels_uniform = {ntype: hidden_channels for ntype in node_types}
            
            self.layers.append(
                MultiRelationAttentionLayer(
                    node_types, edge_types, in_channels_uniform,
                    hidden_channels, num_heads, dropout, edge_attr_dim
                )
            )
        
     
        self.output_layers = nn.ModuleDict()
        for node_type in node_types:
            self.output_layers[node_type] = nn.Sequential(
                nn.Linear(hidden_channels, out_channels),
                nn.LayerNorm(out_channels),
                nn.ReLU()
            )
        
       
        self.prediction_head = LateFusionPredictor(out_channels, dropout)
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict, herb_indices, adr_indices):
       
        h_dict = {ntype: self.input_layers[ntype](x_dict[ntype])
                  for ntype in self.node_types}
        
       
        for layer in self.layers:
            h_dict_agg = layer(h_dict, edge_index_dict, edge_attr_dict)
            
            for ntype in self.node_types:
             
                if ntype in self.node_degrees_dict:
                    degree_tensor = self.node_degrees_dict[ntype]
                    if degree_tensor.device != h_dict_agg[ntype].device:
                        degree_tensor = degree_tensor.to(h_dict_agg[ntype].device)
                    
                    scale = torch.sqrt(degree_tensor + 1).unsqueeze(-1) * torch.sigmoid(self.degree_scale_weight)
                    h_dict_agg[ntype] = h_dict_agg[ntype] * scale
                
               
                eps_weight = 1 + self.eps[ntype]
                h_dict[ntype] = eps_weight * h_dict[ntype] + h_dict_agg[ntype]
        
       
        out_dict = {ntype: self.output_layers[ntype](h_dict[ntype])
                    for ntype in self.node_types}
        
       
        herb_emb = out_dict['herb'][herb_indices]
        adr_emb = out_dict['adr'][adr_indices]
        
       
        if 'herb' in self.node_degrees_dict and 'adr' in self.node_degrees_dict:
            herb_idx_cpu = herb_indices.cpu()
            adr_idx_cpu = adr_indices.cpu()
            
            herb_degree = self.node_degrees_dict['herb'][herb_idx_cpu].unsqueeze(-1).to(herb_emb.device)
            adr_degree = self.node_degrees_dict['adr'][adr_idx_cpu].unsqueeze(-1).to(adr_emb.device)
            
            herb_degree_norm = torch.log(herb_degree + 1.0)
            adr_degree_norm = torch.log(adr_degree + 1.0)
        else:
            herb_degree_norm = torch.zeros(herb_emb.size(0), 1, device=herb_emb.device)
            adr_degree_norm = torch.zeros(adr_emb.size(0), 1, device=adr_emb.device)
        
        
        logits = self.prediction_head(herb_emb, adr_emb, herb_degree_norm, adr_degree_norm)
        predictions = torch.sigmoid(logits)
        
        return predictions.squeeze(-1)
    
    def get_fusion_weights(self):
      
        return F.softmax(self.prediction_head.fusion_weights, dim=0).detach().cpu().numpy()

