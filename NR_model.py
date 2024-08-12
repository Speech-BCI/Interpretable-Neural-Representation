import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import modules.NR_spec_transformer_encoder as spec_transformer_encoder
from scheduling_utils.schedulers import LinearScheduler, CosineScheduler, LinearCosineScheduler
import random
from modules.InfoNCE_loss_criterion import FixedCosineInfoNCE, FixedEuclideanInfoNCE
import os
from modules.specaugment_module import freq_mask, time_mask
import utils as BTS_util
import modules.CNN_encoder_Gausian as CNN_encoder
from modules.abstract_modules.attn_utils import trunc_normal_
import modules.class_balanced_loss as class_balanced_loss
import modules.channel_spatial_attenttion as channel_spatial_attenttion
from modules.spatial_channel_module import get_mne_info


def warmup_cosine_similarity_loss(self, embeddings):
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalization
    cos_sim = torch.matmul(normalized_embeddings, normalized_embeddings.T)

    cos_sim.fill_diagonal_(0)
    loss = cos_sim.abs().mean()

    return loss


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def initialize_linear_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Recording:
    def __init__(self, mne_info, recording_index, study_name, recording_uid):
        self.mne_info = mne_info
        self.recording_index = recording_index
        self.study_name = study_name
        self.recording_uid = recording_uid


class Channel_attn_batch:
    def __init__(self, eeg, recordings):
        self.eeg = eeg
        self._recordings = recordings

class NeuralRepresentationModel(pl.LightningModule):
    def __init__(
            self,
            pretrained_model_path,
            input_channel,
            input_spec_shape,
            n_classes,
            n_class_batches,
            n_classes_trials,
            train_config,
            cnn_config,
            transformer_config,
            embedding_config,
    ):
        super().__init__()
        causal = False
        mask = None
        sparse_topk = None
        use_entmax15 = False
        num_mem_kv = 0
        on_attn = False

        self.save_hyperparameters()
        self.automatic_optimization = False
        self.train_config = train_config
        self.lr = float(self.train_config['lr'])
        self.emb_lr = float(self.train_config['emb_lr'])
        self.warmup_epoch = self.train_config['warmup_epochs']
        self.n_class_batches = n_class_batches
        self.pretrained_model_path = pretrained_model_path
        self.test_embedding = []
        self.test_classes = []
        self.test_attentionmap = []
        self.test_enc_output = []
        self.embedded_dim = transformer_config.embedded_dim
        self.n_classes = n_classes
        self.n_classes_trials = n_classes_trials
        self.relabel_classes = list(range(self.n_classes))
        self.input_channel = input_channel
        self.class_token = transformer_config.class_token
        self.generated_emb_dim = embedding_config.generated_embedding_dim
        self.codebook = nn.Embedding(self.n_classes, self.generated_emb_dim)
        self.pretrained_embedding = embedding_config.pretrained_embedding
        self.cnn_out_channel = cnn_config.output_channels
        if cnn_config.downsampling_rate != 0:
            self.downsampling_rate = cnn_config.downsampling_rate
            self.attn_spec_shape = (input_spec_shape[0], input_spec_shape[1] // (cnn_config.downsampling_rate * (cnn_config.repreated_blocks - 1)))
        else:
            self.downsampling_rate = None
            self.attn_spec_shape = input_spec_shape

        depth = transformer_config.depth
        num_heads = transformer_config.num_heads
        dim_head = transformer_config.dim_head
        talking_heads = transformer_config.talking_heads
        attn_dropout = transformer_config.attn_dropout
        self.patch_size = transformer_config.patch_size
        self.overlap_ratio = transformer_config.overlap_ratio
        self.class_balanced_loss = class_balanced_loss.apply_class_balanced_loss


        if self.pretrained_embedding:
                pretrained_embeddings = np.load('word_embedding_path')
                pretrained_embeddings = torch.tensor(pretrained_embeddings, dtype=torch.float32)
                self.codebook.weight.data.copy_(pretrained_embeddings)
        else:
            ## normal init
            nn.init.zeros_(self.codebook.weight)
            trunc_normal_(self.codebook.weight, std=.02)


        self.ch_input = self.train_config.spatial_attention_out

        # Encoder architecture
        self.enc = CNN_encoder.CNN_feature_encoder(self.ch_input, self.downsampling_rate, cnn_config.repreated_blocks, self.cnn_out_channel)



        # attn architecture
        self.spec_attn = spec_transformer_encoder.AttentionLayers(self.attn_spec_shape, self.embedded_dim, depth, num_heads, dim_head, self.cnn_out_channel, causal, mask,
                                                                  talking_heads, sparse_topk, use_entmax15, num_mem_kv, attn_dropout,
                                                                  attn_dropout, on_attn, self.class_token, self.patch_size, self.overlap_ratio)


        if self.pretrained_model_path is not None:
            checkpoint = torch.load(self.pretrained_model_path)
            ## load pretrained encoder weight
            enc_state_dict = {k.replace('enc.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('enc.')}
            self.enc.load_state_dict(enc_state_dict)
            ## freeze encoder weight
            for param in self.enc.parameters():
                param.requires_grad = False

            ## load pretrained transformer weight
            spec_attn_state_dict = {k.replace('spec_attn.', ''): v for k, v in checkpoint['state_dict'].items() if
                                    k.startswith('spec_attn.')}
            self.spec_attn.load_state_dict(spec_attn_state_dict)

            for param in self.spec_attn.patch_embedding.proj.parameters():
                param.requires_grad = False

        else:
            pass

        self.embedding_checkpoint = train_config.embedding_checkpoint




        self.adaptivepool_1d = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        self.time_mask_augment = freq_mask
        self.freq_mask_augment = time_mask

        self.InfoNCE = FixedCosineInfoNCE()

        self.recording = get_mne_info()
        self.channel_dropout = channel_spatial_attenttion.ChannelDropout(dropout=0.2)
        self.channel_merger = channel_spatial_attenttion.ChannelMerger(chout=self.ch_input)







    def forward(self, x):
        ch_attn = Channel_attn_batch(x, self.recording)
        if self.training:
            x = self.channel_dropout(x, ch_attn)
        x, ch_attn_intermediates = self.channel_merger(x, ch_attn)
        output = self.enc(x)
        if self.class_token:
            attn_output, intermediates = self.spec_attn(output)
            intermediates.post_softmax_attn = intermediates.post_softmax_attn[:, :, 1:, 1:]
            intermediates.pre_softmax_attn = intermediates.pre_softmax_attn[:, :, 1:, 1:]
            output_cls = attn_output[:, 0, :]
        else:
            attn_output, intermediates = self.spec_attn(output)


            cls = self.adaptivepool_1d(attn_output)
            output_cls = cls.view(cls.shape[0], -1)


        if self.training:
            return output_cls, ch_attn_intermediates
        else:
            return output_cls, intermediates.post_softmax_attn.cpu().detach().mean(dim=2), output.detach().mean(dim=1), attn_output



    def loss(self, y_hat, y):
        return self.criteria(y_hat, y)

    def specaugment(self, spec):
        return freq_mask(time_mask(spec))


    def on_train_start(self):
        script_path = __file__
        encoder_script_path = CNN_encoder.__file__

        BTS_util.save_script_log(script_path, log_dir=self.logger.root_dir, file_name = 'model_script')
        BTS_util.save_script_log(encoder_script_path, log_dir=self.logger.root_dir, file_name = 'encoder_script')
        init_codebook = self.codebook.weight.cpu().detach().numpy()
        base_filename = f"{self.logger.root_dir}/init_representation_embedding"
        extension = ".npy"
        np.save(base_filename + extension, init_codebook)




        """
        Initialize warmup or decay of the learning rate (if specified).
        """
        lr = self.lr
        emb_lr = self.emb_lr
        if self.train_config['warmup_epochs'] is not None and self.train_config['decay_epochs'] is not None:
            warmup_step_start = 0
            warmup_step_end = self.train_config['warmup_epochs'] * self.trainer.num_training_batches
            decay_step_end = self.train_config['decay_epochs'] * self.trainer.num_training_batches
            self.attn_warmup_then_decay_lr = LinearCosineScheduler(warmup_step_start, decay_step_end,
                                                              lr, lr / 10, warmup_step_end)

            self.warmup_then_decay_lr = LinearCosineScheduler(warmup_step_start, decay_step_end,
                                                              lr, lr / 10, warmup_step_end)

            ######## emb_lr
            self.warmup_then_decay_emb_lr = LinearCosineScheduler(warmup_step_start, decay_step_end, emb_lr,
                                                                  emb_lr / 10, warmup_step_end)





    def on_train_batch_start(self, batch, batch_index):
        """
        Update lr according to current epoch/batch index
        """
        current_step = (self.current_epoch * self.trainer.num_training_batches) + batch_index

        # lr update
        if self.attn_warmup_then_decay_lr is not None:
            attn_step_lr = self.attn_warmup_then_decay_lr.step(current_step)
        else:
            attn_step_lr = self.train_config['lr']

        if self.warmup_then_decay_emb_lr is not None:
            step_emb_lr = self.warmup_then_decay_emb_lr.step(current_step)
        else:
            step_emb_lr = self.train_config['emb_lr']

        for optimizer_idx, _optimizer in enumerate(self.trainer.optimizers):
            if optimizer_idx == 0:
                for o, g in enumerate(_optimizer.param_groups):
                    if o == 0:
                        g['lr'] = attn_step_lr
                    elif o == 1:
                        g['lr'] = attn_step_lr
                    elif o == 2:
                        ## cls token learning rate
                        g['lr'] = attn_step_lr

            elif optimizer_idx == 1:
                for g in _optimizer.param_groups:
                    g['lr'] = attn_step_lr
            elif optimizer_idx == 2:
                if self.current_epoch > self.warmup_epoch:
                    for g in _optimizer.param_groups:
                        g['lr'] = step_emb_lr
                else:
                    for g in _optimizer.param_groups:
                        g['lr'] = self.lr
            elif optimizer_idx == 3:
                for g in _optimizer.param_groups:
                    g['lr'] = attn_step_lr





        self.log('attn_lr', attn_step_lr, on_step=True, on_epoch=True, prog_bar=True)
        self.log('emb_lr', step_emb_lr, on_step=True, on_epoch=True, prog_bar=True)




    def training_step(self, batch, batch_idx):
        x, y = batch
        attn_optimizer, enc_optimizer, embedding_optimizer, spatial_attn_optimizer  = self.optimizers()

        output_cls, ch_attn_intermediates = self(x)




        y_label = y.to(y.device)

        reference_embedding = self.codebook(y_label)


        all_indices = torch.arange(self.n_classes).to(y.device)
        all_indices = all_indices.repeat(y_label.shape[0], 1)


        negative_indices = torch.zeros_like(all_indices[:, :self.n_classes-1])
        for i in range(y_label.shape[0]):
            negative_indices[i] = torch.cat((all_indices[i, :y_label[i]], all_indices[i, y_label[i] + 1:]))


        negative_embedding = torch.zeros((y_label.shape[0], self.n_classes-1, self.generated_emb_dim)).to(y.device)
        for i in range(y_label.shape[0]):
            negative_embedding[i] = self.codebook(negative_indices[i])



        ref = F.normalize(output_cls, p=2, dim=1)
        pos = F.normalize(reference_embedding, p=2, dim=1)
        neg = F.normalize(negative_embedding, p=2, dim=2)


        _, pos_loss, neg_loss = self.InfoNCE(ref, pos, neg)


        ###### InfoNCE Loss

        beta = 0.999
        aligned_pos_loss = self.class_balanced_loss(pos_loss, y_label, self.n_classes_trials.to(y.device), beta)
        aligned_neg_loss = self.class_balanced_loss(neg_loss, y_label, self.n_classes_trials.to(y.device), beta)
        aligned_cl_loss = aligned_pos_loss.mean() + aligned_neg_loss.mean()


        _, _, codebook_neg_loss = self.InfoNCE(pos, ref, neg)
        aligned_codebook_neg_loss = self.class_balanced_loss(codebook_neg_loss, y_label, self.n_classes_trials.to(y.device), beta)
        aligned_codebook_neg_loss = aligned_codebook_neg_loss.mean()

        tr_loss = aligned_cl_loss + aligned_codebook_neg_loss

        if self.current_epoch > self.warmup_epoch:


            attn_optimizer.zero_grad()
            enc_optimizer.zero_grad()
            embedding_optimizer.zero_grad()
            spatial_attn_optimizer.zero_grad()
            self.manual_backward(tr_loss)

            attn_optimizer.step()
            enc_optimizer.step()
            embedding_optimizer.step()
            spatial_attn_optimizer.step()
        else:
            aligned_codebook_loss = warmup_cosine_similarity_loss(self, self.codebook.weight)

            attn_optimizer.zero_grad()
            enc_optimizer.zero_grad()
            spatial_attn_optimizer.zero_grad()
            self.manual_backward(tr_loss)
            attn_optimizer.step()
            enc_optimizer.step()
            spatial_attn_optimizer.step()

            embedding_optimizer.zero_grad()
            self.manual_backward(aligned_codebook_loss)
            embedding_optimizer.step()


        self.log('train_loss', tr_loss, on_step=False, on_epoch=True, prog_bar=True)

        return tr_loss


    def on_train_epoch_end(self):
        if self.current_epoch % self.embedding_checkpoint != (self.embedding_checkpoint-1) or self.current_epoch == 0:
            self.trainer.limit_val_batches = 0
        else:
            self.trainer.limit_val_batches = 1.0


    def validation_step(self, batch, batch_idx):
        x, y = batch

        output_cls, _, _, _ = self(x)
        y_label = y.to(y.device)

        reference_embedding = self.codebook(y_label)

        all_indices = torch.arange(self.n_classes).to(y.device)
        all_indices = all_indices.repeat(y_label.shape[0], 1)

        negative_indices = torch.zeros_like(all_indices[:, :self.n_classes - 1])
        for i in range(y_label.shape[0]):
            negative_indices[i] = torch.cat((all_indices[i, :y_label[i]], all_indices[i, y_label[i] + 1:]))



        negative_embedding = torch.zeros((y_label.shape[0], self.n_classes - 1, self.generated_emb_dim)).to(y.device)
        for i in range(y_label.shape[0]):
            negative_embedding[i] = self.codebook(negative_indices[i])



        ref = F.normalize(output_cls, p=2, dim=1)
        pos = F.normalize(reference_embedding, p=2, dim=1)
        neg = F.normalize(negative_embedding, p=2, dim=2)

        _, pos_loss, neg_loss = self.InfoNCE(ref, pos, neg)

        beta = 0.999
        aligned_pos_loss = self.class_balanced_loss(pos_loss, y_label, self.n_classes_trials.to(y.device), beta)
        aligned_neg_loss = self.class_balanced_loss(neg_loss, y_label, self.n_classes_trials.to(y.device), beta)
        aligned_cl_loss = aligned_pos_loss.mean() + aligned_neg_loss.mean()

        _, _, codebook_neg_loss = self.InfoNCE(pos, ref, neg)
        aligned_codebook_neg_loss = self.class_balanced_loss(codebook_neg_loss, y_label,
                                                             self.n_classes_trials.to(y.device), beta)
        aligned_codebook_neg_loss = aligned_codebook_neg_loss.mean()


        val_loss = aligned_cl_loss + aligned_codebook_neg_loss



        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_pos_loss', pos_loss.mean(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_neg_loss', neg_loss.mean(), on_step=False, on_epoch=True, prog_bar=True)

        return val_loss




    def on_validation_epoch_end(self):
        if (self.current_epoch % (self.embedding_checkpoint * 10)) == 0:
            all_codebook = self.codebook.weight.cpu().detach().numpy()


    def test_step(self, batch, batch_idx):
        x, y = batch
        output_cls, attentionmap, enc_output, _  = self(x)
        y_label = y.to(y.device)

        reference_embedding = self.codebook(y_label)

        all_indices = torch.arange(self.n_classes).to(y.device)

        all_indices = all_indices.repeat(y_label.shape[0], 1)

        negative_indices = torch.zeros_like(all_indices[:, :self.n_classes - 1])
        for i in range(y_label.shape[0]):
            negative_indices[i] = torch.cat((all_indices[i, :y_label[i]], all_indices[i, y_label[i] + 1:]))


        negative_embedding = torch.zeros((y_label.shape[0], self.n_classes - 1, self.generated_emb_dim)).to(y.device)
        for i in range(y_label.shape[0]):
            negative_embedding[i] = self.codebook(negative_indices[i])


        ref = F.normalize(output_cls, p=2, dim=1)
        pos = F.normalize(reference_embedding, p=2, dim=1)
        neg = F.normalize(negative_embedding, p=2, dim=2)

        _, pos_loss, neg_loss = self.InfoNCE(ref, pos, neg)

        beta = 0.999
        aligned_pos_loss = self.class_balanced_loss(pos_loss, y_label, self.n_classes_trials.to(y.device), beta)
        aligned_neg_loss = self.class_balanced_loss(neg_loss, y_label, self.n_classes_trials.to(y.device), beta)
        aligned_cl_loss = aligned_pos_loss.mean() + aligned_neg_loss.mean()

        _, _, codebook_neg_loss = self.InfoNCE(pos, ref, neg)
        aligned_codebook_neg_loss = self.class_balanced_loss(codebook_neg_loss, y_label,
                                                             self.n_classes_trials.to(y.device), beta)
        aligned_codebook_neg_loss = aligned_codebook_neg_loss.mean()



        test_loss = aligned_cl_loss + aligned_codebook_neg_loss



        self.test_embedding.append(output_cls)
        self.test_classes.append(y)
        self.test_attentionmap.append(attentionmap)
        self.test_enc_output.append(enc_output)

        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True)


        return test_loss


    def on_test_epoch_end(self):

        ## make results folder
        test_result_dir = f"{self.logger.root_dir}/test_results"
        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)

        current_epoch = 0

        test_attentionmap = torch.cat(self.test_attentionmap, dim=0)
        np.save(f"{self.logger.root_dir}/test_attentionmap_{current_epoch}", test_attentionmap.cpu().numpy())
        test_enc_output = torch.cat(self.test_enc_output, dim=0)
        np.save(f"{self.logger.root_dir}/test_enc_output_{current_epoch}", test_enc_output.cpu().numpy())

        ## class of all_trials_embedding
        test_classes = torch.cat(self.test_classes, dim=0).cpu().detach().numpy()
        test_classes = test_classes.astype(int)
        i = 0
        base_filename = f"{test_result_dir}/test_classes_{current_epoch}_{i}"
        extension = ".npy"
        while os.path.exists(base_filename + extension):
            i += 1
            base_filename = f"{test_result_dir}/test_{current_epoch}_classes_{i}"
        np.save(base_filename + extension, test_classes)


        ## all_trials_embedding
        test_embedding = torch.cat(self.test_embedding, dim=0).cpu().detach().numpy()
        i = 0
        base_filename = f"{test_result_dir}/test_{current_epoch}_embedding_{i}"
        extension = ".npy"
        while os.path.exists(base_filename + extension):
            i += 1
            base_filename = f"{test_result_dir}/test_{current_epoch}_embedding_{i}"
        np.save(base_filename + extension, test_embedding)


        ## codebook embedding
        all_embeddings = self.codebook.weight.cpu().detach().numpy()
        i = 0
        base_filename = f"{test_result_dir}/test_{current_epoch}_representation_embedding_{i}"
        extension = ".npy"
        while os.path.exists(base_filename + extension):
            i += 1
            base_filename = f"{test_result_dir}/test_{current_epoch}_representation_embedding_{i}"
        np.save(base_filename + extension, all_embeddings)


        self.test_embedding = []
        self.test_classes = []
        self.test_attentionmap = []
        self.test_enc_output = []


    def predict_step(self, batch, batch_idx):
        x, y = batch
        output_cls, attentionmap, enc_output, _  = self(x)

        output_cls = F.normalize(output_cls, p=2, dim=1)


        y_label = y.to(y.device)

        codebook = self.codebook.weight
        codebook = F.normalize(codebook, p=2, dim=1)


        similarity_matrix = torch.zeros((output_cls.shape[0], codebook.shape[0]))
        for i, output_vec in enumerate(output_cls):
            for j, codebook_vec in enumerate(codebook):
                similarity_matrix[i, j] = F.cosine_similarity(output_vec.unsqueeze(0), codebook_vec.unsqueeze(0), dim=1)

        top10_indices = torch.topk(similarity_matrix, codebook.shape[0], dim=1)[1]

        rank_cosine_similarity_matrix = torch.zeros((output_cls.shape[0], 10))


        return top10_indices, y_label, rank_cosine_similarity_matrix








    def configure_optimizers(self):

        def split_decay_groups(named_modules: list, named_parameters: list,
                               whitelist_weight_modules: tuple[torch.nn.Module, ...],
                               blacklist_weight_modules: tuple[torch.nn.Module, ...],
                               wd: float):
            """
            reference https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/vqvae.py
            separate out all parameters to those that will and won't experience regularizing weight decay
            """

            decay = set()
            no_decay = set()
            cls_token_decay = set()
            for mn, m in named_modules:
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
                    elif pn in ['pre_softmax_proj', 'post_softmax_proj', 'pos_embed', 'code_book']:
                        # The specified parameters will not be decayed
                        no_decay.add(fpn)
                    elif pn in ['cls_token']:
                        # The specified parameters will not be decayed
                        cls_token_decay.add(fpn)

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in named_parameters}
            inter_params = decay & no_decay
            union_params = decay | no_decay | cls_token_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
            assert len(
                param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params),)

            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": wd},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
                {"params": [param_dict[pn] for pn in sorted(list(cls_token_decay))], "weight_decay": wd},
            ]
            return optim_groups

        lr = float(self.train_config['lr'])
        betas = [float(b) for b in self.train_config['betas']]
        eps = float(self.train_config['eps'])
        weight_decay = float(self.train_config['weight_decay'])

        attn_params = split_decay_groups(
            named_modules=list(self.spec_attn.named_modules()),
            named_parameters=list(self.spec_attn.named_parameters()),
            whitelist_weight_modules=(torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding),
            blacklist_weight_modules=(torch.nn.GroupNorm, torch.nn.LayerNorm),
            wd=weight_decay
        )


        attn_optimizer = torch.optim.AdamW(attn_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


        enc_optimizer = torch.optim.AdamW(self.enc.parameters(),
                                        lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        embedding_optimizer = torch.optim.AdamW(self.codebook.parameters(),
                                                lr = lr, betas=betas, eps=eps, weight_decay=weight_decay)

        spatial_attn_optimizer = torch.optim.AdamW(self.channel_merger.parameters(),
                                                   lr = lr, betas=betas, eps=eps, weight_decay=weight_decay)


        return [attn_optimizer, enc_optimizer, embedding_optimizer, spatial_attn_optimizer]



