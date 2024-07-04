import os
import torch
import numpy as np
import pandas as pd
import os
import shutil
import datetime
import torch.nn.functional as F
from scipy.signal import butter, filtfilt



class Bandpass_filtering:
    @staticmethod
    def bandpass_filter(data, sr, freq_range):
        b, a = butter(N=4, Wn=np.array(freq_range) / (sr / 2), btype='bandpass')
        filtered_data = np.empty_like(data)
        for trial in range(data.shape[0]):
            for channel in range(data.shape[1]):
                filtered_data[trial, channel, :] = filtfilt(b, a, data[trial, channel, :])
        return filtered_data


class Highpass_filtering:
    @staticmethod
    def highpass_filter(data, sr, cutoff_freq):
        b, a = butter(N=4, Wn=cutoff_freq / (sr / 2), btype='highpass')
        filtered_data = np.empty_like(data)

        for trial in range(data.shape[0]):
            for channel in range(data.shape[1]):
                filtered_data[trial, channel, :] = filtfilt(b, a, data[trial, channel, :])

        return filtered_data


class Val_list:
    def __init__(self):
        self.train_indices_list = []
        self.val_indices_list = []
        self.test_indices_list = []
        self.test_acc_list = []
        self.test_ema_acc_list = []
        self.best_train_indices = []
        self.best_val_indices = []
        self.best_test_indices = []


    def __setattr__(self, key, value):
        if hasattr(self, key):
            getattr(self, key).append(value)
        else:
            super().__setattr__(key, value)

    def reset(self):
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                value.clear()

    def add_to_list(self, key, value):
        if hasattr(self, key):
            getattr(self, key).append(value)
        else:
            super().__setattr__(key, [value])

def make_save_folder(path):
    folder_name = path
    models_path = os.path.join(folder_name, 'models')
    if not os.path.exists(folder_name) and not os.path.exists(models_path):
        os.makedirs(folder_name)
    elif os.path.exists(folder_name) and not os.path.exists(models_path):
        pass
    else:
        i = 2
        while True:
            new_folder_name = f"{folder_name}_new_ver{i}"
            models_path = os.path.join(new_folder_name, 'models')
            if not os.path.exists(new_folder_name) and not os.path.exists(models_path):
                os.makedirs(new_folder_name)
                folder_name = new_folder_name
                print(f"A new folder has been created: {new_folder_name}")
                break
            elif os.path.exists(new_folder_name) and not os.path.exists(models_path):
                folder_name = new_folder_name
                print(f"Folder already exists: {new_folder_name}")
                break

            i += 1

    plot_folder = f"{folder_name}/plot_folder"

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    return folder_name, plot_folder

def configure_device(device):
    if device.startswith("gpu"):
        if not torch.cuda.is_available():
            raise Exception(
                "CUDA support is not available on your platform. Re-run using CPU or TPU mode"
            )
        gpu_id = device.split(":")[-1]
        if gpu_id == "":
            # Use all GPU's
            gpu_id = -1
        gpu_id = [int(id) for id in gpu_id.split(",")]

        gpu_num = ["cuda:" + str(num) for num in np.array(gpu_id)]
        return gpu_num, gpu_id
    return device


def load_best_loss_model(path):

    files = os.listdir(path)
    files = [f for f in files if f.endswith('.ckpt')]


    # files.sort(key=lambda f: float(f.split('val_loss_dataloader_idx_0=')[1].split('.ckpt')[0]), reverse=False)
    # val_best = files[0].split('val_loss_dataloader_idx_0=')[1].split('.ckpt')[0]
    files.sort(key=lambda f: float(f.split('val_loss=')[1].split('.ckpt')[0]), reverse=False)
    val_best = files[0].split('val_loss=')[1].split('.ckpt')[0]
    val_acc_files = [f for f in files if val_best in f]

    max_epoch = -1
    max_epoch_file = None

    for f in val_acc_files:
        epoch_str = f.split('=')[1].split('_')[0]
        epoch = int(epoch_str)
        if epoch > max_epoch:
            max_epoch = epoch
            max_epoch_file = f

    if max_epoch_file is not None:
        print(f"The file with the lowest 'val_Loss=' and the highest 'epoch=': {max_epoch_file}")
    else:
        print("No file has been found.")

    file_path = f"{path}/{max_epoch_file}"


    return file_path

def load_pretrained_model(path, args):
    pretrained_model_path = f'{path}/attn_vqcl_allwords_{args.train_sub}_{args.task}/models/'
    pretrained_model_path = load_best_loss_model(pretrained_model_path)
    print("pretrained_model_path: ", pretrained_model_path)
    return pretrained_model_path




def save_script_log(script_path, log_dir='logs', file_name = 'model_script'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"log_{timestamp}_{file_name}.py"
    log_file_path = os.path.join(log_dir, log_file_name)

    shutil.copy(script_path, log_file_path)
    print(f"Script logged as {log_file_path}")


def analyze_and_save_stat_results(task, epoch, t_statistic, p_value, wilcoxon_statistic, wilcoxon_p_value, nn_embedding, stat_analysis_dir, file_name, num_sheet = 'sheet1'):
    def determine_groups(p_value, statistic, is_t_test=True):
        if 0.01 < p_value <= 0.05:
            if (statistic > 0 and is_t_test) or (statistic < 0 and not is_t_test):
                greater_group = 'semantic group'
                lesser_group = 'phonological group'
            else:
                greater_group = 'phonological group'
                lesser_group = 'semantic group'
        elif 0.001 < p_value <= 0.01:
            if (statistic > 0 and is_t_test) or (statistic < 0 and not is_t_test):
                greater_group = 'good semantic group'
                lesser_group = 'phonological group'
            else:
                greater_group = 'good phonological group'
                lesser_group = 'semantic group'
        elif p_value <= 0.001:
            if (statistic > 0 and is_t_test) or (statistic < 0 and not is_t_test):
                greater_group = 'best semantic group'
                lesser_group = 'phonological group'
            else:
                greater_group = 'best phonological group'
                lesser_group = 'semantic group'
        else:
            greater_group = 'None'
            lesser_group = 'None'
        return greater_group, lesser_group

    greater_group, lesser_group = determine_groups(p_value, t_statistic, is_t_test=True)

    wilcoxon_greater_group, wilcoxon_lesser_group = determine_groups(wilcoxon_p_value, wilcoxon_statistic, is_t_test=False)

    # Excel 파일 경로 결정
    if nn_embedding:
        excel_file = os.path.join(stat_analysis_dir, 'nn_stat_results.xlsx')
        task = 'nn_' + task
    else:
        excel_file = os.path.join(stat_analysis_dir, file_name)


    data = {
        'Num': [epoch],
        'Task': task,
        't_statistic': [t_statistic],
        'p_value': [p_value],
        'greater_group': [greater_group],
        'lesser_group': [lesser_group],
        'wilcoxon_statistic': [wilcoxon_statistic],
        'wilcoxon_p_value': [wilcoxon_p_value],
        'wilcoxon_greater_group': [wilcoxon_greater_group],
        'wilcoxon_lesser_group': [wilcoxon_lesser_group]
    }

    if not os.path.exists(excel_file):
        df = pd.DataFrame(data)
        df.to_excel(excel_file, sheet_name=num_sheet,index=False)
    else:
        df = pd.read_excel(excel_file)
        new_df = pd.DataFrame(data)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_excel(excel_file, sheet_name=num_sheet, index=False)





def load_best_loss_epoch(path):

    files = os.listdir(path)
    files = [f for f in files if f.endswith('.ckpt')]


    files.sort(key=lambda f: float(f.split('val_loss=')[1].split('.ckpt')[0]), reverse=False)
    val_best = files[0].split('val_loss=')[1].split('.ckpt')[0]
    val_acc_files = [f for f in files if val_best in f]

    max_epoch = -1
    max_epoch_file = None

    for f in val_acc_files:
        epoch_str = f.split('=')[1].split('_')[0]
        epoch = int(epoch_str)
        if epoch > max_epoch:
            max_epoch = epoch
            max_epoch_file = f


    file_path = f"{path}/{max_epoch_file}"

    return file_path, max_epoch



def plot_class_arrange(embedding, labels, str_labels):

    class_distributions = {}
    class_embedding = []

    class_labels = []
    for embedding_, label in zip(embedding, labels):
        if label not in class_distributions:
            class_distributions[label] = {"embedding": []}
        class_distributions[label]["embedding"].append(embedding_)

    for label, dist in class_distributions.items():
        embedding_array = np.array(dist["embedding"])
        class_embedding.append(embedding_array)
        class_labels.append(label)

    class_labels = np.array(class_labels)


    sorted_indices = np.argsort(class_labels)
    sorted_class_embedding = [class_embedding[i] for i in sorted_indices]
    class_embedding = sorted_class_embedding

    arranged_labels = []
    for idx in range(len(class_labels)):
        arranged_labels.append(str_labels[class_labels[np.where(class_labels == idx)[0][0]]])
    arranged_labels = np.array(arranged_labels)

    return class_embedding, arranged_labels


def calculate_codebook_similarity(codebook, embedding_labels, class_name, arranged_class_name):
    indices = {item: [i for i, x in enumerate(arranged_class_name) if x == item] for item in class_name}
    indices_only = [index for sublist in indices.values() for index in sublist]
    indices_only = np.array(indices_only)

    remapped_label = np.zeros_like(embedding_labels)

    ## arrange of each classes label
    for original, remapped in enumerate(indices_only):
        remapped_label[embedding_labels == original] = remapped


    class_embeddings, arranged_labels = plot_class_arrange(codebook, remapped_label, arranged_class_name)


    class_embeddings = np.array(class_embeddings)
    class_embeddings = class_embeddings.reshape(class_embeddings.shape[0], -1)

    class_embeddings = torch.from_numpy(class_embeddings)

    similarity_matrix = torch.zeros((len(class_embeddings), len(class_embeddings)))
    for i in range(len(class_embeddings)):
        similarity_matrix[i] = F.cosine_similarity(class_embeddings[i], class_embeddings, dim=1)

    similarity_matrix = similarity_matrix.cpu().detach().numpy()

    return similarity_matrix, arranged_labels

