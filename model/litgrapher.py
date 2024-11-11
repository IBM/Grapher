import torch
import pytorch_lightning as pl
from misc.utils import compute_loss, decode_text, decode_graph, compute_scores
import torch.nn.functional as F
import torch.nn as nn
import os
from model.grapher import Grapher


def sorted_ls(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    return list(sorted(os.listdir(path), key=mtime))


class FocalLoss(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma, weight=None):
        super(FocalLoss, self).__init__(weight)
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):

        # input: batch_size x vocab_size x d1 x ... x dn x seq_len
        # target: batch_size x d1 x ... x dn x seq_len
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = ((1 - p) ** self.gamma * ce_loss)

        return focal_loss

class LitGrapher(pl.LightningModule):
    def __init__(self,
                 transformer_class,
                 transformer_name,
                 tokenizer,
                 cache_dir,
                 max_nodes,
                 max_edges,
                 edges_as_classes,
                 default_seq_len_edge,
                 num_classes,
                 dropout_rate,
                 num_layers,
                 vocab_size,
                 bos_token_id,
                 eos_token_id,
                 nonode_id,
                 noedge_id,
                 node_sep_id,
                 noedge_cl,
                 edge_classes,
                 focal_loss_gamma,
                 eval_dir,
                 lr,
                 ):
        super().__init__()
        self.save_hyperparameters()

        model = Grapher(transformer_class=transformer_class,
                        transformer_name=transformer_name,
                        cache_dir=cache_dir,
                        max_nodes=max_nodes,
                        edges_as_classes=edges_as_classes,
                        node_sep_id=node_sep_id,
                        default_seq_len_edge=default_seq_len_edge,
                        num_classes=num_classes,
                        dropout_rate=dropout_rate,
                        num_layers=num_layers,
                        vocab_size=vocab_size,
                        bos_token_id=bos_token_id)

        self.model = model
        self.criterion = {'ce': nn.CrossEntropyLoss(reduction='none'), 'focal': FocalLoss(focal_loss_gamma)}
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.transformer_name = transformer_name
        self.edges_as_classes = edges_as_classes
        self.edge_classes = edge_classes
        self.focal_loss_gamma = focal_loss_gamma
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.node_sep_id = node_sep_id
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.noedge_cl = noedge_cl
        self.nonode_id = nonode_id
        self.noedge_id = noedge_id
        self.eval_dir=eval_dir
        self.lr = lr

    def training_step(self, batch, batch_idx):

        # target_nodes: batch_size X seq_len_node
        # target_edges: num_nodes X num_nodes X batch_size X seq_len_edge [FULL]
        # target_edges: batch_size X num_nodes X num_nodes [CLASSES]
        text_input_ids, text_input_attn_mask, target_nodes, target_nodes_mask, target_edges = batch

        # logits_nodes: batch_size X seq_len_node X vocab_size
        # logits_edges: num_nodes X num_nodes X batch_size X seq_len_edge X vocab_size [FULL]
        # logits_edges: num_nodes X num_nodes X batch_size X num_classes [CLASSES]
        logits_nodes, logits_edges= self.model(text_input_ids,
                                               text_input_attn_mask,
                                               target_nodes,
                                               target_nodes_mask,
                                               target_edges)

        loss = compute_loss(self.criterion, logits_nodes, logits_edges, target_nodes,
                            target_edges, self.edges_as_classes, self.focal_loss_gamma)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True, batch_size=text_input_ids.size(0))

        return loss

    def eval_step(self, batch, batch_idx, split):

        iteration = self.global_step
        rank = self.global_rank

        text_input_ids, text_input_attn_mask, target_nodes, target_nodes_mask, target_edges = batch

        logits_nodes, seq_nodes, logits_edges, seq_edges = self.model.sample(text_input_ids, text_input_attn_mask)

        text_dec = decode_text(self.tokenizer, text_input_ids, self.bos_token_id, self.eos_token_id)

        TB_str = []

        dec_target = decode_graph(self.tokenizer, self.edge_classes, target_nodes, target_edges, self.edges_as_classes,
                                  self.node_sep_id, self.max_nodes, self.noedge_cl, self.noedge_id,
                                  self.bos_token_id, self.eos_token_id)

        dec_pred = decode_graph(self.tokenizer, self.edge_classes, seq_nodes, seq_edges, self.edges_as_classes,
                                self.node_sep_id, self.max_nodes, self.noedge_cl, self.noedge_id,
                                self.bos_token_id, self.eos_token_id)

        if batch_idx == 0:
            for b_i in range(len(text_dec)):
                # ---- ground truth ----
                gt = '<br/>'.join('-->'.join(tri) for tri in dec_target[b_i])

                # ---- predicted  -------
                pr = '<br/>'.join('-->'.join(tri) for tri in dec_pred[b_i])

                strng = f'{b_i}<br/>' + text_dec[b_i] + '<br/>' \
                        + '-' * 40 + 'target' + '-' * 40 + '<br/>' + gt + '<br/>' \
                        + '-' * 40 + 'predicted' + '-' * 20 + '<br/>' + pr + '<br/>'
                TB_str.append(strng)

        decodes = {'text_dec': text_dec, 'dec_target': dec_target, 'dec_pred': dec_pred}

        for i, tb_str in enumerate(TB_str):
            self.logger.experiment.add_text(f'{split}_{rank}/{i}', tb_str, iteration)

        return decodes

    def eval_epoch_end(self, outputs, split):

        iteration = self.global_step
        rank = self.global_rank

        dec_target_all = []
        dec_pred_all = []

        for out in outputs:
            dec_target_all += out['dec_target']
            dec_pred_all += out['dec_pred']

        # make sure number of paths is smaller than 10
        dec_pred_all = [tr[:10] for tr in dec_pred_all]
        dec_target_all = [tr[:10] for tr in dec_target_all]

        # hack to avoid crashing the program if evaluation fails
        try:
            scores = compute_scores(dec_pred_all, dec_target_all, iteration, self.eval_dir, split, rank)
        except Exception:
            return

        for k, v in scores.items():
            self.logger.experiment.add_scalar(f'{split}_score/{k}', v, global_step=iteration)

        self.log_dict(scores)

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'valid')

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, 'test')

    def on_validation_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, 'valid')

    def test_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), self.lr)
        return optimizer