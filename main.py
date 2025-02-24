
import os
import json
import hydra
import logging
from omegaconf import DictConfig

from tqdm import tqdm

import torch
import statistics
from torch.utils.data import DataLoader
from continuum.metrics import Logger

from continual_clip import utils
from continual_clip.models import load_model
from continual_clip.datasets import build_cl_scenarios
from continual_clip.models import ClassIncremental

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.colors import Normalize


@hydra.main(config_path=None, config_name=None, version_base="1.1") 
def continual_clip(cfg: DictConfig) -> None:

    cfg.workdir = utils.get_workdir(path=os.getcwd())  
    cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)  

    utils.save_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))  
    model  = load_model(cfg, device)   
    eval_dataset_img, eval_dataset_audio, classes_names = build_cl_scenarios(
        cfg, is_train=False, transforms=model.transforms
    )
    train_dataset_img, train_dataset_audio, train_classes_names = build_cl_scenarios(
        cfg, is_train=True, transforms=model.transforms
    )
    model.classes_names = classes_names     

    acc_list = []
    acc_list1 = []
    similarity_dict = {}
    forgetting_dict = {}
    bwt_dict = {}
    fwt_dict = {}
    metric_logger = Logger(list_subsets=["test"])  


    for task_id, _ in enumerate(eval_dataset_img):  
        logging.info(f"Evaluation for task {task_id} has started.")
        model.adaptation(task_id, cfg, train_dataset_img, train_dataset_audio, train_classes_names)  

        eval_loader_img = DataLoader(eval_dataset_img[:task_id + 1], batch_size=64)  
        eval_loader_audio = DataLoader(eval_dataset_audio[:task_id + 1], batch_size=64)
        
        for (inputs, targets, task_ids), (inputs_audio, targets_audio, task_ids_audio) in tqdm(zip(eval_loader_img, eval_loader_audio)): 
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_audio, targets_audio = inputs_audio.to(device), targets_audio.to(device)   
            outputs, task_prototype_dict, nmi_fused_image, nmi_fused_audio, image_features = model(inputs, inputs_audio, task_id)
            metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")  

        acc_list.append(100 * metric_logger.accuracy)  
        acc_list1.append(metric_logger.accuracy)  


        if task_id > 0:
            current_embedding_text = model.embeddings_dict[task_id]
            similarity_text = model.cosine_similarity(current_embedding_text, previous_embedding_text).item()
            similarity_text = similarity_text / 2 + 0.5
            previous_embedding_text = current_embedding_text

            current_embedding_image = task_prototype_dict[task_id]
            similarity_image = model.cosine_similarity(current_embedding_image, previous_embedding_image).item()
            similarity_image = similarity_image / 2 + 0.5
            previous_embedding_image = current_embedding_image
        else:
            similarity_text = 1.0
            similarity_image = 1.0
            previous_embedding_text = model.embeddings_dict[0]
            previous_embedding_image = task_prototype_dict[0]
      
        similarity = similarity_text / 2 + similarity_image / 2

        similarity_dict[task_id] = similarity
        forgetting_dict[task_id] = round(100 * metric_logger.forgetting, 4)
        bwt_dict[task_id] = round(metric_logger.backward_transfer, 4)  
        fwt_dict[task_id] = round(metric_logger.forward_transfer, 4)

        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'task_similarity': round(similarity_image, 4),
                'acc': round(100 * metric_logger.accuracy, 2),
                'avg_acc': round(100 * metric_logger.average_incremental_accuracy, 2),
                'forgetting': round(100 * metric_logger.forgetting, 4),
                'acc_per_task': [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task],
                'bwt': round(100 * metric_logger.backward_transfer, 2),
                'fwt': round(100 * metric_logger.forward_transfer, 2)
            }) + '\n')
            metric_logger.end_task()    

    with open(cfg.log_path, 'a+') as f:
        f.write(json.dumps({
            'last': round(acc_list[-1], 2), 
            'avg': round(statistics.mean(acc_list), 2)
        }) + '\n')

    acc_avg = round(statistics.mean(acc_list), 2) 
    acc_avg1 = round(statistics.mean(acc_list1), 4) 

    for_weighted_avg = sum((1 - similarity_dict[task_id]) * (100 - forgetting_dict[task_id])  for task_id in similarity_dict) / len(similarity_dict)
    metric1 =  acc_avg / 2 + for_weighted_avg / 2   

    bwt_fwt_avg = sum((bwt_dict[task_id] + fwt_dict[task_id])  for task_id in bwt_dict) / len(bwt_dict)
    metric2 =  (acc_avg1 / 2 + bwt_fwt_avg / 2) / 2 + (nmi_fused_image.item() * 0.5 + nmi_fused_audio.item() * 0.5) / 2  

    with open(cfg.log_path, 'a+') as f:
        f.write(json.dumps({
            'metric1': round(metric1, 4),
            'metric2': round(metric2, 4)
            }) + '\n')    


if __name__ == "__main__":
    continual_clip()