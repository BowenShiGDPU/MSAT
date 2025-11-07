
import os
from pathlib import Path


class ModelConfig:
    
    HIDDEN_CHANNELS = 576
    OUT_CHANNELS = 72
    NUM_LAYERS = 3
    NUM_HEADS = 8
    DROPOUT = 0.18
    EDGE_ATTR_DIM = 6
    
   
    USE_GATED_EDGE_ENCODER = True      
    USE_BOTTLENECK_TRANSFORM = True   
    USE_LATE_FUSION = True             
    USE_DEGREE_FEATURES = True         
    USE_EPSILON = True                 
    USE_DEGREE_SCALING = True          
    USE_GLOBAL_SOFTMAX = True         


class TrainingConfig:
    
   
    NUM_EPOCHS = 1000
    PATIENCE = 100
    
    
    LEARNING_RATE = 0.0004
    WEIGHT_DECAY = 1e-5
    OPTIMIZER = 'AdamW'
    
  
    GRADIENT_CLIP = 1.0
    BATCH_SIZE = 512
    
   
    N_FOLDS = 10
    TRAIN_VAL_SPLIT = 0.9  
    
   
    SCHEDULER_MODE = 'max'
    SCHEDULER_FACTOR = 0.6
    SCHEDULER_PATIENCE = 15
    
  
    RANDOM_STATE = 42


class DataConfig:
    
  
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = os.path.join(PROJECT_ROOT, 'experiments_data_clean_final')
    
  
    N_FOLDS = 10
    RANDOM_SEED = 42
    
 
    INDUCTIVE_MODE = True

