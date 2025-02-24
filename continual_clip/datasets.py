
import os
import torch.nn as nn

from continuum import ClassIncremental, InstanceIncremental
from continuum.datasets import (
    CIFAR100, ImageNet100, TinyImageNet200, ImageFolderDataset, Core50
)
from .utils import get_dataset_class_names

from continuum.datasets import _ContinuumDataset
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from typing import Tuple, Union, Optional
from continuum.tasks import TaskType


class ClassroomData_img(_ContinuumDataset):  
    def __init__(self, data_path: str, train: bool = True):
        super(ClassroomData_img, self).__init__(data_path, train)
        self.data_path = data_path  
        self.train = train
        self.class_to_idx = {}

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        #############
        # data_path = os.path.join(self.data_path, "image", 'train' if self.train else 'test')
        data_path = os.path.join("/home/server_medical3/cyk/MoE-Adapters1/data/classroom_data", "image", 'train' if self.train else 'test')
        class_names = os.listdir(data_path)
        class_names.sort() 
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

        if not self.train:
            x_test = []
            y_test = []
            for class_name in class_names:
                class_path = os.path.join(data_path, class_name)
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    x_test.append(img_path)
                    y_test.append(self.class_to_idx[class_name])
            return np.array(x_test), np.array(y_test), None

        x_train = []
        y_train = []
        for class_name in class_names:
            class_path = os.path.join(data_path, class_name)
            img_names = os.listdir(class_path)
            x_train.extend([os.path.join(class_path, img_name) for img_name in img_names])
            y_train.extend([self.class_to_idx[class_name]] * len(img_names))
        return np.array(x_train), np.array(y_train), None



class ClassroomData_audio(_ContinuumDataset):   
    def __init__(self, data_path: str, train: bool = True):
        super(ClassroomData_audio, self).__init__(data_path, train)
        self.data_path = data_path  
        ###################
        # self.audio_data_path = os.path.join(self.data_path, "audio")
        self.audio_data_path = os.path.join("/home/server_medical3/cyk/MoE-Adapters1/data/classroom_data", "audio")  
        self.train = train
        self.class_to_idx = {}

    @property
    def data_type(self) -> TaskType:
        return TaskType.AUDIO

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        ###################
        # data_path = os.path.join(self.data_path, "image", 'train' if self.train else 'test') 
        data_path = os.path.join("/home/server_medical3/cyk/MoE-Adapters1/data/classroom_data", "image", 'train' if self.train else 'test')   
        audio_data_path = self.audio_data_path  
        class_names = os.listdir(data_path) 
        class_names.sort()  
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}  
        if not self.train:
            x_test = []
            y_test = []
            for class_name in class_names:
                class_path = os.path.join(data_path, class_name)  
                for img_name in os.listdir(class_path):
                    img_name_part = "wav_" + img_name.split("_", 2)[-1]  
                    audio_path = os.path.join(audio_data_path, img_name_part.replace('.jpg', '.wav'))  
                    x_test.append(audio_path)  
                    y_test.append(self.class_to_idx[class_name])
            return np.array(x_test), np.array(y_test), None  

        x_train = []
        y_train = []
        for class_name in class_names:
            class_path = os.path.join(data_path, class_name)  
            img_names = os.listdir(class_path)  
            x_train.extend([os.path.join(audio_data_path, ("wav_" + img_name.split("_", 2)[-1]).replace('.jpg', '.wav')) for img_name in img_names])
            y_train.extend([self.class_to_idx[class_name]] * len(img_names))
        return np.array(x_train), np.array(y_train), None  
 
 
class ImageNet19_img(_ContinuumDataset):
    def __init__(self, data_path: str, train: bool = True):
        super(ImageNet19_img, self).__init__(data_path, train)
        self.data_path = data_path
        self.train = train
        self.class_to_idx = {}

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        data_path = os.path.join(self.data_path, "image", 'train' if self.train else 'test')  
        class_names = os.listdir(data_path)
        class_names.sort()  
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

        if not self.train:
            x_test = []
            y_test = []
            for class_name in class_names:
                class_path = os.path.join(data_path, class_name)  
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)  
                    x_test.append(img_path)
                    y_test.append(self.class_to_idx[class_name])
            return np.array(x_test), np.array(y_test), None

        x_train = []
        y_train = []
        for class_name in class_names:
            class_path = os.path.join(data_path, class_name)
            img_names = os.listdir(class_path)
            x_train.extend([os.path.join(class_path, img_name) for img_name in img_names])
            y_train.extend([self.class_to_idx[class_name]] * len(img_names))
        return np.array(x_train), np.array(y_train), None
    
    
class ImageNet19_audio(_ContinuumDataset):  
    def __init__(self, data_path: str, train: bool = True):
        super(ImageNet19_audio, self).__init__(data_path, train)
        self.data_path = data_path  
        self.audio_data_path = os.path.join(self.data_path, "audio")  
        self.train = train
        self.class_to_idx = {}

    @property
    def data_type(self) -> TaskType:
        return TaskType.AUDIO

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        data_path = os.path.join(self.data_path, "image", 'train' if self.train else 'test')
        audio_data_path = self.audio_data_path  
        class_names = os.listdir(data_path)
        class_names.sort()  
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

        if not self.train:
            x_test = []
            y_test = []
            for class_name in class_names:
                class_path = os.path.join(data_path, class_name)  
                for img_name in os.listdir(class_path):
                    audio_name = class_name + '.wav'
                    audio_path = os.path.join(audio_data_path, audio_name)  
                    x_test.append(audio_path)  
                    y_test.append(self.class_to_idx[class_name])
            return np.array(x_test), np.array(y_test), None

        x_train = []
        y_train = []
        for class_name in class_names:
            class_path = os.path.join(data_path, class_name)  
            img_names = os.listdir(class_path)
            x_train.extend([os.path.join(audio_data_path, (class_name + '.wav')) for img_name in img_names])
            y_train.extend([self.class_to_idx[class_name]] * len(img_names))
        return np.array(x_train), np.array(y_train), None

    

class ImageNet27_img(_ContinuumDataset):
    def __init__(self, data_path: str, train: bool = True):
        super(ImageNet27_img, self).__init__(data_path, train)
        self.data_path = data_path
        self.train = train
        self.class_to_idx = {}

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        data_path = os.path.join(self.data_path, "image", 'train' if self.train else 'test')  
        class_names = os.listdir(data_path)
        class_names.sort()  
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

        if not self.train:
            x_test = []
            y_test = []
            for class_name in class_names:
                class_path = os.path.join(data_path, class_name)  
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)  
                    x_test.append(img_path)
                    y_test.append(self.class_to_idx[class_name])
            return np.array(x_test), np.array(y_test), None

        x_train = []
        y_train = []
        for class_name in class_names:
            class_path = os.path.join(data_path, class_name)
            img_names = os.listdir(class_path)
            x_train.extend([os.path.join(class_path, img_name) for img_name in img_names])
            y_train.extend([self.class_to_idx[class_name]] * len(img_names))
        return np.array(x_train), np.array(y_train), None
    
    
class ImageNet27_audio(_ContinuumDataset):
    def __init__(self, data_path: str, train: bool = True):
        super(ImageNet27_audio, self).__init__(data_path, train)
        self.data_path = data_path  
        self.audio_data_path = os.path.join(self.data_path, "audio")  
        self.train = train
        self.class_to_idx = {}

    @property
    def data_type(self) -> TaskType:
        return TaskType.AUDIO

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        data_path = os.path.join(self.data_path, "image", 'train' if self.train else 'test')
        audio_data_path = self.audio_data_path  
        class_names = os.listdir(data_path)
        class_names.sort()  
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

        if not self.train:
            x_test = []
            y_test = []
            for class_name in class_names:
                class_path = os.path.join(data_path, class_name)  
                for img_name in os.listdir(class_path):
                    audio_name = class_name + '.wav'
                    audio_path = os.path.join(audio_data_path, audio_name)  
                    x_test.append(audio_path)  
                    y_test.append(self.class_to_idx[class_name])
            return np.array(x_test), np.array(y_test), None

        x_train = []
        y_train = []
        for class_name in class_names:
            class_path = os.path.join(data_path, class_name)
            img_names = os.listdir(class_path)

            x_train.extend([os.path.join(audio_data_path, (class_name + '.wav')) for img_name in img_names])
            y_train.extend([self.class_to_idx[class_name]] * len(img_names))
        return np.array(x_train), np.array(y_train), None
    
    

def get_dataset(cfg, is_train, transforms=None):
    if cfg.dataset == "imagenet19":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)    
        dataset_img = ImageNet19_img(
            data_path=data_path, 
            train=is_train, 
        )
        dataset_audio = ImageNet19_audio(
            data_path=data_path, 
            train=is_train, 
        )
        classes_names = [
            "cock", "hen", "chickadee", "tree frog",
            "Otterhound",  "Egyptian cat","fly",
            "cricket", "pig", "big-horn sheep" ,
            "airliner", "bullet train" ,"chainsaw",
            "computer keyboard", "digital clock", "computer mouse",
            "vacuum cleaner", "wall clock", "washing machine"
        ]


    elif cfg.dataset == "classroom_data":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)    
        dataset_img = ClassroomData_img(
            data_path=data_path, 
            train=is_train, 
        )
        dataset_audio = ClassroomData_audio(
            data_path=data_path, 
            train=is_train, 
        )
        classes_names = [
            "analyzing", "blackboard_wiping", "blackboard_writing", "discussing", "drinking",
            "eating", "gathering_up_bag", "handsup", "listening", "listening_to_music",
            "picking_up_computers", "reading", "relaxing", "reviewing", "scratching_head",
            "setting_equipment", "sleeping", "speaking", "standing" ,"student_demonstrating",
            "taking_bag", "taking_bottle", "taking_off", "taking_photos" ,"talking",
            "teaching", "unknown", "using_computers", "using_pad" ,"using_phone",
            "walking", "writing", "yawning"
        ]


    elif cfg.dataset == "imagenet27":
        data_path = os.path.join(cfg.dataset_root, cfg.dataset)    
        dataset_img = ImageNet27_img(
            data_path=data_path, 
            train=is_train, 
        )
        dataset_audio = ImageNet27_audio(
            data_path=data_path, 
            train=is_train, 
        )
        classes_names = [
            "cock", "hen", "chickadee", "tree frog","Otterhound", "Egyptian cat",
            "fly","cricket", "pig", "big-horn sheep" ,"airliner", 
            "bullet train", "can opener", "chainsaw", "church","computer keyboard",
            "digital clock", "fire screen", "computer mouse","toilet seat", "vacuum cleaner", 
            "wall clock", "washbasin", "washing machine", "water bottle", "water jug", "sandbar"
        ]
    
    else:
        ValueError(f"'{cfg.dataset}' is a invalid dataset.")

    return dataset_img, dataset_audio, classes_names


def build_cl_scenarios(cfg, is_train, transforms) -> nn.Module:

    dataset_img, dataset_audio, classes_names = get_dataset(cfg, is_train)

    if cfg.scenario == "class":  
        scenario_img = ClassIncremental(
            dataset_img,
            initial_increment=cfg.initial_increment,
            increment=cfg.increment,
            transformations=transforms.transforms, 
            class_order=cfg.class_order,
        )
        scenario_audio = ClassIncremental(
            dataset_audio,
            initial_increment=cfg.initial_increment,
            increment=cfg.increment,
            transformations=transforms.transforms, 
            class_order=cfg.class_order,
        )

    else:
        ValueError(f"You have entered `{cfg.scenario}` which is not a defined scenario, " 
                    "please choose from {{'class', 'domain', 'task-agnostic'}}.")

    return scenario_img, scenario_audio, classes_names


