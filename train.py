
import os
import sys
import json
import time
import copy
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef
)
from scipy import stats
from scipy.stats import spearmanr

import psutil
try:
    import GPUtil
except:
    GPUtil = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.feature_extractor import FeatureExtractor
from MSAT_TCMFS_Final.model import MSATTCMFSFinal
from MSAT_TCMFS_Final.config import ModelConfig, TrainingConfig, DataConfig


def train_one_epoch(model, data, optimizer, train_edges, train_labels, device):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict,
                train_edges[0].to(device), train_edges[1].to(device))
    
    if torch.isnan(out).any():
        return float('nan')
    
    loss = F.binary_cross_entropy(out, train_labels.to(device))
    
    if torch.isnan(loss):
        return float('nan')
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.GRADIENT_CLIP)
    optimizer.step()
    
    return loss.item()


def evaluate(model, data, herb_indices, adr_indices, labels, device):
    model.eval()
    
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict,
                   torch.LongTensor(herb_indices).to(device),
                   torch.LongTensor(adr_indices).to(device))
        
        pred_probs = torch.nan_to_num(out, nan=0.5).cpu().numpy()
        pred_labels = (pred_probs >= 0.5).astype(int)
    
    return {
        'precision': float(precision_score(labels, pred_labels, zero_division=0)),
        'recall': float(recall_score(labels, pred_labels, zero_division=0)),
        'f1': float(f1_score(labels, pred_labels, zero_division=0)),
        'auc': float(roc_auc_score(labels, pred_probs)),
        'auprc': float(average_precision_score(labels, pred_probs)),
        'mcc': float(matthews_corrcoef(labels, pred_labels))
    }, pred_probs


def train_single_fold(fold_idx, device):
    print(f"\n{'='*80}")
    print(f"Fold {fold_idx + 1}/{TrainingConfig.N_FOLDS}")
    print(f"{'='*80}")
    
    fold_start_time = time.time()
    
   
    process = psutil.Process()
    cpu_before = process.cpu_percent(interval=0.1)
    mem_before = process.memory_info().rss / 1024 / 1024
    
    gpu_mem_before = 0
    if GPUtil:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_mem_before = gpus[0].memoryUsed
        except:
            pass
    
   
    extractor = FeatureExtractor(data_dir=DataConfig.DATA_DIR)
    data = extractor.get_graph_data()
    train_data, test_data = extractor.load_fold_data(fold_idx)
    
    print(f"  Train samples: {len(train_data['labels'])}")
    print(f"  Test samples:  {len(test_data['labels'])}")
    
  
    test_pairs = set(zip(test_data['herb_indices'].tolist(), test_data['adr_indices'].tolist()))
    
    for edge_index_key in [
        ('herb', 'causes', 'adr'),
        ('adr', 'rev_causes', 'herb')
    ]:
        edge_index = data[edge_index_key].edge_index
        keep = []
        for i in range(edge_index.size(1)):
            if edge_index_key[0] == 'herb':
                pair = (int(edge_index[0,i]), int(edge_index[1,i]))
            else:
                pair = (int(edge_index[1,i]), int(edge_index[0,i]))
            keep.append(pair not in test_pairs)
        
        data[edge_index_key].edge_index = edge_index[:, torch.tensor(keep)]
        if hasattr(data[edge_index_key], 'edge_attr') and data[edge_index_key].edge_attr is not None:
            data[edge_index_key].edge_attr = data[edge_index_key].edge_attr[torch.tensor(keep)]
    
    print(f"  Excluded {len(test_pairs)} test edges")
    
   
    node_degrees_dict = {}
    for ntype in data.node_types:
        degree = torch.zeros(data[ntype].x.size(0))
        for edge_type, edge_index in data.edge_index_dict.items():
            if edge_type[2] == ntype:
                degree += torch.bincount(edge_index[1], minlength=data[ntype].x.size(0)).float()
        node_degrees_dict[ntype] = degree
    
    data = data.to(device)
    
  
    n_train = len(train_data['labels'])
    indices = np.random.permutation(n_train)
    n_val = int(n_train * 0.1)
    
    train_edges = torch.LongTensor([
        train_data['herb_indices'][indices[n_val:]],
        train_data['adr_indices'][indices[n_val:]]
    ])
    val_edges = (train_data['herb_indices'][indices[:n_val]], train_data['adr_indices'][indices[:n_val]])
    train_labels = torch.FloatTensor(train_data['labels'][indices[n_val:]])
    val_labels = train_data['labels'][indices[:n_val]]
    
    print(f"  Train/Val split: {len(train_labels)}/{len(val_labels)}")
    
   
    model = MSATTCMFSFinal(
        node_types=list(data.node_types),
        edge_types=list(data.edge_types),
        in_channels_dict={ntype: data[ntype].x.size(1) for ntype in data.node_types},
        hidden_channels=ModelConfig.HIDDEN_CHANNELS,
        out_channels=ModelConfig.OUT_CHANNELS,
        num_layers=ModelConfig.NUM_LAYERS,
        num_heads=ModelConfig.NUM_HEADS,
        dropout=ModelConfig.DROPOUT,
        edge_attr_dim=ModelConfig.EDGE_ATTR_DIM,
        node_degrees_dict=node_degrees_dict
    ).to(device)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
   
    optimizer = AdamW(
        model.parameters(),
        lr=TrainingConfig.LEARNING_RATE,
        weight_decay=TrainingConfig.WEIGHT_DECAY
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=TrainingConfig.SCHEDULER_MODE,
        factor=TrainingConfig.SCHEDULER_FACTOR,
        patience=TrainingConfig.SCHEDULER_PATIENCE
    )
    
   
    train_start_time = time.time()
    best_val_auc = 0
    best_epoch = 0
    patience = 0
    best_state = None
    train_history = {'epochs': [], 'train_loss': [], 'val_auc': []}
    
    for epoch in range(TrainingConfig.NUM_EPOCHS):
        train_loss = train_one_epoch(model, data, optimizer, train_edges, train_labels, device)
        
        if np.isnan(train_loss):
            print(f"  ERROR: NaN loss at epoch {epoch+1}")
            break
        
        val_metrics, _ = evaluate(model, data, val_edges[0], val_edges[1], val_labels, device)
        scheduler.step(val_metrics['auc'])
        
        train_history['epochs'].append(epoch + 1)
        train_history['train_loss'].append(train_loss)
        train_history['val_auc'].append(val_metrics['auc'])
        
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch + 1
            patience = 0
            best_state = copy.deepcopy(model.state_dict())
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}: val_auc={val_metrics['auc']:.4f} *")
        else:
            patience += 1
            if patience >= TrainingConfig.PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    train_time = time.time() - train_start_time
    
    
    if best_state:
        model.load_state_dict(best_state)
        
        
        Path('MSAT_TCMFS_Final/saved_models').mkdir(parents=True, exist_ok=True)
        torch.save(best_state, f'MSAT_TCMFS_Final/saved_models/best_model_fold{fold_idx}.pt')
    
   
    cpu_after = process.cpu_percent(interval=0.1)
    mem_after = process.memory_info().rss / 1024 / 1024
    
    gpu_mem_after = 0
    if GPUtil:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_mem_after = gpus[0].memoryUsed
        except:
            pass
    
   
    test_metrics, test_preds = evaluate(model, data, test_data['herb_indices'], 
                                        test_data['adr_indices'], test_data['labels'], device)
    
    fold_time = time.time() - fold_start_time
    
    print(f"\n  Fold {fold_idx+1} Results:")
    print(f"    Precision: {test_metrics['precision']:.4f}")
    print(f"    Recall:    {test_metrics['recall']:.4f}")
    print(f"    F1:        {test_metrics['f1']:.4f}")
    print(f"    AUC:       {test_metrics['auc']:.4f}")
    print(f"    AUPR:      {test_metrics['auprc']:.4f}")
    print(f"    MCC:       {test_metrics['mcc']:.4f}")
    print(f"    Time:      {fold_time:.1f}s")
    
    return {
        'fold': fold_idx,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'test_metrics': test_metrics,
        'predictions': {
            'y_true': test_data['labels'].tolist(),
            'y_score': test_preds.tolist(),
            'y_pred': (test_preds >= 0.5).astype(int).tolist()
        },
        'training_history': train_history,
        'resource_usage': {
            'train_time_seconds': train_time,
            'total_time_seconds': fold_time,
            'cpu_percent_avg': (cpu_before + cpu_after) / 2,
            'memory_mb_peak': mem_after,
            'gpu_memory_mb_before': gpu_mem_before,
            'gpu_memory_mb_after': gpu_mem_after
        }
    }


def run_10fold_cv():
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Data: {DataConfig.DATA_DIR}\n")
    
    experiment_start = time.time()
    results = []
    best_overall_auc = -1
    best_overall_fold = -1
    best_overall_state = None
    
    for fold in range(TrainingConfig.N_FOLDS):
        fold_result = train_single_fold(fold, device)
        results.append(fold_result)
        
        if fold_result['test_metrics']['auc'] > best_overall_auc:
            best_overall_auc = fold_result['test_metrics']['auc']
            best_overall_fold = fold
           
            state_path = f'MSAT_TCMFS_Final/saved_models/best_model_fold{fold}.pt'
            if os.path.exists(state_path):
                best_overall_state = torch.load(state_path, map_location='cpu')
    
    total_experiment_time = time.time() - experiment_start
    
   
    print(f"\n{'='*80}")
    print("Overall Results (10-Fold CV)")
    print("="*80)
    
    overall_metrics = {}
    for metric in ['precision', 'recall', 'f1', 'auc', 'auprc', 'mcc']:
        values = [r['test_metrics'][metric] for r in results]
        overall_metrics[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': values
        }
        print(f"  {metric.upper():12s}: {overall_metrics[metric]['mean']:.4f} ± {overall_metrics[metric]['std']:.4f}")
    
    
    avg_train_time = np.mean([r['resource_usage']['train_time_seconds'] for r in results])
    avg_total_time = np.mean([r['resource_usage']['total_time_seconds'] for r in results])
    avg_cpu = np.mean([r['resource_usage']['cpu_percent_avg'] for r in results])
    avg_gpu = np.mean([r['resource_usage']['gpu_memory_mb_after'] for r in results if r['resource_usage']['gpu_memory_mb_after'] > 0])
    
    print(f"\nResource Usage:")
    print(f"  Avg train time: {avg_train_time:.1f}s")
    print(f"  Avg total time: {avg_total_time:.1f}s")
    print(f"  Avg CPU: {avg_cpu:.1f}%")
    if avg_gpu > 0:
        print(f"  Avg GPU: {avg_gpu:.1f} MB")
    
    
    print(f"\nStatistical Significance:")
    
    auc_values = overall_metrics['auc']['values']
    
   
    t_stat_random, p_value_random = stats.ttest_1samp(auc_values, 0.5)
    print(f"  vs Random (0.5): t={t_stat_random:.4f}, p={p_value_random:.2e}")
    
   
    auc_auprc_corr, _ = spearmanr(overall_metrics['auc']['values'], overall_metrics['auprc']['values'])
    print(f"  AUC-AUPRC correlation: {auc_auprc_corr:.4f}")
    
    
    Path('MSAT_TCMFS_Final/results').mkdir(parents=True, exist_ok=True)
    
    summary = {
        'model_name': 'MSAT-TCMFS-Final',
        'model_info': {
            'architecture': 'MSAT with 3 core innovations',
            'innovations': [
                'Gated Edge Encoder',
                'Bottleneck Output Transform',
                'Late Fusion with Degree Features'
            ],
            'reference': 'Based on ablation study, uses Global Softmax'
        },
        'data_config': {
            'data_dir': 'experiments_data_clean_final',
            'n_folds': TrainingConfig.N_FOLDS,
            'random_seed': DataConfig.RANDOM_SEED
        },
        'model_config': {
            'hidden_channels': ModelConfig.HIDDEN_CHANNELS,
            'out_channels': ModelConfig.OUT_CHANNELS,
            'num_layers': ModelConfig.NUM_LAYERS,
            'num_heads': ModelConfig.NUM_HEADS,
            'dropout': ModelConfig.DROPOUT,
            'edge_attr_dim': ModelConfig.EDGE_ATTR_DIM
        },
        'training_config': {
            'learning_rate': TrainingConfig.LEARNING_RATE,
            'weight_decay': TrainingConfig.WEIGHT_DECAY,
            'num_epochs': TrainingConfig.NUM_EPOCHS,
            'patience': TrainingConfig.PATIENCE,
            'gradient_clip': TrainingConfig.GRADIENT_CLIP
        },
        'overall_metrics': overall_metrics,
        'statistical_tests': {
            't_test_vs_random': {
                't_statistic': float(t_stat_random),
                'p_value': float(p_value_random)
            },
            'auc_auprc_correlation': float(auc_auprc_corr)
        },
        'fold_results': results,
        'resource_usage_avg': {
            'train_time_seconds': avg_train_time,
            'total_time_seconds': avg_total_time,
            'cpu_percent': avg_cpu,
            'gpu_memory_mb': avg_gpu if avg_gpu > 0 else 0
        },
        'experiment_info': {
            'total_experiment_time_seconds': total_experiment_time,
            'avg_best_epoch': np.mean([r['best_epoch'] for r in results]),
            'timestamp': datetime.now().isoformat()
        }
    }
    
    with open('MSAT_TCMFS_Final/results/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[SAVED] MSAT_TCMFS_Final/results/summary.json")
    
  
    if best_overall_state is not None:
        torch.save(best_overall_state, 'MSAT_TCMFS_Final/saved_models/best_model_for_prediction.pt')
        print(f"[SAVED] MSAT_TCMFS_Final/saved_models/best_model_for_prediction.pt (from fold {best_overall_fold})")
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    run_10fold_cv()

