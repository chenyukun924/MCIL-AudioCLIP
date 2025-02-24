from omegaconf import DictConfig
from tqdm import tqdm
import torch.nn.functional as F

import clip.clip as clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import get_class_ids_per_task, get_class_names, batch, merge_we_router, wise_we, moving_avg, l2_loss, \
    virtual_vocab, distillation
import copy

from .cc import conceptual_captions

from . import utils
import os
import random
import numpy as np
from .dynamic_dataset import DynamicDataset

from torchvision.transforms import ToPILImage, ToTensor

from collections import OrderedDict

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .text_templates import CLASSROOM_TEMPLATES

from esresnet import ESResNeXtFBSP
from clip.model import CLIP
from typing import Tuple
from typing import Union
from typing import Optional

import math

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class FusionNet(nn.Module):  
    def __init__(self, input_dim, output_dim=512, dropout_rate=0.5):
        super(FusionNet, self).__init__()
        self.image_weight = nn.Parameter(torch.ones(1))
        self.audio_weight = nn.Parameter(torch.ones(1))
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim * 2, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
            )
        ])
        self.threshold = 0.8  

    def calculate_correlation(self, image_feature, audio_feature):
        vx = image_feature - torch.mean(image_feature, dim=1, keepdim=True)
        vy = audio_feature - torch.mean(audio_feature, dim=1, keepdim=True)
        correlation = torch.sum(vx * vy, dim=1) / (torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vy ** 2, dim=1)))
        return torch.mean(correlation)

    def forward(self, image_feature, audio_feature):
        correlation = self.calculate_correlation(image_feature, audio_feature)
        if correlation < self.threshold:
            x = image_feature
        else:
            x = torch.cat((image_feature * self.image_weight, audio_feature * self.audio_weight), dim=1)
            Q = image_feature 
            K = x
            V = x
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
            attention_weights = nn.Softmax(dim=-1)(attention_scores)
            attention_output = torch.matmul(attention_weights, V)
            x = attention_output 
            for layer in self.fusion_layers:
                x = layer(x)
        return x       



class MyLoss(nn.Module): 

    def mutual_information(self, x, y):
        joint_prob = F.softmax(x * y, dim=-1)
        marginal_prob_x = F.softmax(x, dim=-1)
        marginal_prob_y = F.softmax(y, dim=-1)
        mi = (joint_prob * (torch.log(joint_prob) - torch.log(marginal_prob_x) - torch.log(marginal_prob_y))).sum()
        return -mi

    def forward(self, cfg, image_features, audio_features, fuse_features, logits_fused, logits_img, image_labels):
        cosine_similarity = F.cosine_similarity(image_features.unsqueeze(1), image_features.unsqueeze(0), dim=-1)
        similarity_weights = cosine_similarity / 2 + 0.5
        loss_ce = F.cross_entropy(logits_fused, image_labels, reduction='none', label_smoothing=cfg.ls)
        sum_weights_i = similarity_weights.sum(dim=1, keepdim=True)
        weighted_loss = similarity_weights * loss_ce.unsqueeze(1)
        normalized_loss = weighted_loss.sum(dim=1) / sum_weights_i.squeeze(1)
        loss_cwcl = normalized_loss.mean()

        loss_mi_image = self.mutual_information(fuse_features, image_features)
        loss_mi_audio = self.mutual_information(fuse_features, audio_features)
        loss_mi = loss_mi_image + loss_mi_audio

        total_loss = 0.7 * loss_cwcl + 0.3 * loss_mi   

        return total_loss


def normalized_mutual_information(x, y):   
    joint_prob = F.softmax(x * y, dim=-1)
    marginal_prob_x = F.softmax(x, dim=-1)
    marginal_prob_y = F.softmax(y, dim=-1)
    mi = (joint_prob * (torch.log(joint_prob) - torch.log(marginal_prob_x) - torch.log(marginal_prob_y))).sum()

    entropy_x = -(marginal_prob_x * torch.log(marginal_prob_x)).sum()
    entropy_y = -(marginal_prob_y * torch.log(marginal_prob_y)).sum()

    nmi = mi / torch.sqrt(entropy_x * entropy_y)
    return nmi


class AudioCLIP(CLIP):  
    def __init__(self, 
                 clip_model_name, 
                 device,
                 embed_dim: int = 1024,
                 # vision
                 image_resolution: int = 224,
                 vision_layers: Union[Tuple[int, int, int, int], int] = (3, 4, 6, 3),
                 vision_width: int = 64,
                 vision_patch_size: Optional[int] = None,
                 # text
                 context_length: int = 77,
                 vocab_size: int = 49408,
                 transformer_width: int = 512,
                 transformer_heads: int = 8,
                 transformer_layers: int = 12,
                 # audio
                 n_fft: int = 2048,
                 hop_length: Optional[int] = 561,
                 win_length: Optional[int] = 1654,
                 window: Optional[str] = 'blackmanharris',
                 normalized: bool = True,
                 onesided: bool = True,
                 spec_height: int = -1,
                 spec_width: int = -1,
                 apply_attention: bool = True,
                 multilabel: bool = True,
                 pretrained: Union[bool, str] = True):

        super(AudioCLIP, self).__init__(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            context_length=context_length,
            vocab_size=vocab_size,
            transformer_width=transformer_width,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers
        )

        self.audio_encoder = ESResNeXtFBSP(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            normalized=normalized,
            onesided=onesided,
            spec_height=spec_height,
            spec_width=spec_width,
            num_classes=embed_dim,
            apply_attention=apply_attention,
            pretrained=True
        )
       
        self.clip_model, self.transforms, _ = clip.load(clip_model_name, device, jit=False)
        self.image_encoder = self.clip_model.encode_image 
        self.text_encoder = self.clip_model.encode_text
        self.fusion_net = FusionNet(input_dim=1024)  
        self.fc = nn.Linear(1024, 512)  


    def forward(self, image, audio, text, taskid, is_train):  
        global global_taskid, global_is_train
        global_taskid = taskid
        global_is_train = is_train
        
        task_prototype_dict = {}  
        nmi_fused_image = None
        nmi_fused_audio = None

        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        average_prototype = torch.mean(image_features, dim=0)
        task_prototype_dict[taskid] = average_prototype  
        
        audio_features = self.audio_encoder(audio)
        audio_features = self.fc(audio_features)
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

        text_features = self.text_encoder(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        fused_features = self.fusion_net(image_features, audio_features)
        
        nmi_fused_image = normalized_mutual_information(image_features,fused_features)  
        nmi_fused_audio = normalized_mutual_information(audio_features,fused_features)  

        logits_fused = self.logit_scale * fused_features @ text_features.t()

        return logits_fused, task_prototype_dict, nmi_fused_image, nmi_fused_audio, image_features



class ClassIncremental(nn.Module):  
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None   
        _, self.transforms, _ = clip.load(cfg.model_name, device=device, jit=jit)
        self.model = AudioCLIP(clip_model_name=cfg.model_name, device=device)
        self.ref_model = None
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))  
        self.current_class_names = []
        self.text_tokens = None
        self.dynamic_dataset = DynamicDataset(cfg)
        self.embeddings_dict = {}  


    def cosine_similarity(self, embedding1, embedding2):   
        return F.cosine_similarity(embedding1, embedding2, dim=0)
    

    def forward(self, image, audio, task_id):  
        with torch.no_grad():
            logits_fused, task_prototype_dict, nmi_fused_image, nmi_fused_audio, image_features = self.model(image, audio, self.text_tokens, task_id, is_train=False) 
        probs = logits_fused.softmax(dim=-1)  
        return probs, task_prototype_dict, nmi_fused_image, nmi_fused_audio, image_features 


    def adaptation(self, task_id, cfg, train_dataset_img, train_dataset_audio, train_classes_names):  
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])  
        self.text_tokens = clip.tokenize(   
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)  

        embeddings = self.model.text_encoder(self.text_tokens)

        total_embedding = torch.zeros_like(embeddings[0])  
        for embedding in embeddings:
            total_embedding += embedding  
        average_embedding = total_embedding / len(embeddings)  
        self.embeddings_dict[task_id] = average_embedding

        if cfg.method != "zeroshot":  
            self.train(task_id, cfg, train_dataset_img, train_dataset_audio, train_classes_names)


    def train(self, task_id, cfg, train_dataset_img, train_dataset_audio, train_classes_names):  
        ### loading dataset
        train_loader_img = DataLoader(train_dataset_img[task_id:task_id + 1],
                                  batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=8)
        train_loader_audio = DataLoader(train_dataset_audio[task_id:task_id + 1],
                                  batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=8)
        
        train_iter_img = iter(train_loader_img)  
        train_iter_audio = iter(train_loader_audio)  

        EPOCH = 1
        num_batches = len(train_loader_img)
        total_iterations = EPOCH * num_batches

        ### whole-model
        exclude_params_name = ["logit_scale"]

        for k, v in self.model.named_parameters(): 
            if "adaptmlp" not in k and "router" not in k and "noise" not in k and "fusion_net" not in k :  
                v.requires_grad = False

        params = [
            v for k, v in self.model.named_parameters() if "adaptmlp" in k or "router" in k or "noise" in k or "fusion_net" in k  
        ]

        # print('========trainable params============', params)

        logit_scale = self.model.logit_scale

        # optimizer
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = utils.cosine_lr(
            optimizer, cfg.lr, 30, total_iterations
        )

        self.model = self.model.to(self.device)

        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        all_texts = []
        # Using multiple text templates to ensure textual diversity during training
        for single_template in CLASSROOM_TEMPLATES:   
            x = [single_template.replace("{}", name) for name in classnames]
            all_texts.extend([clip.tokenize(p) for p in x])
        all_texts = torch.cat(all_texts)


        # start training
        self.model.train()

        for iteration in tqdm(range(total_iterations + 1)):
            scheduler(iteration)
            try:
                inputs, targets, task_ids = next(train_iter_img)  
                inputs_audio, targets_audio, task_ids = next(train_iter_audio)  
            except:
                train_iter_img = iter(train_loader_img)
                train_iter_audio = iter(train_loader_audio)
                inputs, targets, task_ids = next(train_iter_img)  
                inputs_audio, targets_audio, task_ids = next(train_iter_audio)  

            inputs, inputs_audio, targets = inputs.to(self.device), inputs_audio.to(self.device), targets.to(self.device)

            image_features = self.model.image_encoder(inputs)  
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_text_features = self.model.text_encoder(all_texts.to(self.device))
            text_embeddings = all_text_features.view(len(CLASSROOM_TEMPLATES), len(classnames), -1).mean(dim=0)   
            
            with torch.no_grad():
                audio_features = self.model.audio_encoder(inputs_audio)  
            audio_features = self.model.fc(audio_features)
            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

            features = self.model.fusion_net(image_features, audio_features)
            features = features / features.norm(dim=-1, keepdim=True)

            logit_scale = self.model.logit_scale
            logits_fused = logit_scale * features @ text_embeddings.t()
            logits_img = logit_scale * image_features @ text_embeddings.t()
            

            if cfg.dataset == "tinyimagenet" and task_id != 0:
                shift = 100 + (task_id - 1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "classroom_data" and task_id != 0:   
                shift = cfg.initial_increment + (task_id - 1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "imagenet27" and task_id != 0:   
                shift = cfg.initial_increment + (task_id - 1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "imagenet19" and task_id != 0:   
                shift = cfg.initial_increment + (task_id - 1) * cfg.increment
                targets -= shift
            else:
                shift = task_id * cfg.increment
                targets -= shift

            criterion = MyLoss()
            loss = criterion(cfg, image_features, audio_features, features, logits_fused, logits_img, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.model.eval()


class DomainIncremental(nn.Module):
    pass


class TaskAgnostic(nn.Module):
    pass


def load_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.

    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.

    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        return ClassIncremental(cfg, device)
    elif cfg.scenario == "domain":
        return DomainIncremental(cfg, device)
    elif cfg.scenario == "task-aganostic":
        return TaskAgnostic(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario, 
            Please choose from ['class', "domain', 'task-agnostic']
        """)
