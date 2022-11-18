import pytorch_lightning as pl
from corpusreader.benchmark_reader import Benchmark
from corpusreader.benchmark_reader import select_files
import os
import json
import unicodedata as ud
import networkx as nx
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch


class GraphDataModule(pl.LightningDataModule):

    def __init__(self,
                 tokenizer_class,
                 tokenizer_name,
                 cache_dir,
                 data_path,
                 dataset,
                 batch_size,
                 num_data_workers,
                 max_nodes,
                 max_edges,
                 edges_as_classes):
        super().__init__()

        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_name, cache_dir=cache_dir)
        self.tokenizer.add_tokens('__no_node__')
        self.tokenizer.add_tokens('__no_edge__')
        self.tokenizer.add_tokens('__node_sep__')

        self.batch_size = batch_size
        self.num_data_workers = num_data_workers
        self.data_path = data_path
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.edges_as_classes = edges_as_classes
        self.output_path = os.path.join(data_path, 'processed')
        os.makedirs(self.output_path, exist_ok=True)
        self.dataset = dataset

    def prepareWebNLG(self):

        splits = ['train', 'dev', 'test']

        for split in splits:

            text_file = os.path.join(self.output_path, f'{split}.text')
            graph_file = os.path.join(self.output_path, f'{split}.graph')
            if os.path.exists(text_file) and os.path.exists(graph_file):
                continue

            b = Benchmark()
            if split == 'test':
                files = [(os.path.join(self.data_path, split), 'semantic-parsing-test-data-with-refs-en.xml')]
            else:
                files = select_files(os.path.join(self.data_path, split))
            b.fill_benchmark(files)
            b.b2json(self.output_path, f'{split}.json')
            D = json.load(open(os.path.join(self.output_path, f'{split}.json')))

            normalize = lambda text: ud.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

            triples_to_write = []
            text_to_write = []
            edge_classes_to_write = []
            for ind, entry in enumerate(D['entries']):
                triples = entry[str(ind + 1)]['modifiedtripleset']
                proc_triples = []
                for triple in triples:
                    obj, rel, sub = triple['object'], triple['property'], triple['subject']
                    obj = normalize(obj.strip('\"').replace('_', ' '))
                    sub = normalize(sub.strip('\"').replace('_', ' '))
                    proc_triples.append(f'__subject__ {sub} __predicate__ {rel} __object__ {obj}')
                    if split == 'train':
                        edge_classes_to_write.append(rel)

                merged_triples = ' '.join(proc_triples)

                proc_lexs = [normalize(l['lex']) for l in entry[str(ind + 1)]['lexicalisations']]

                for lex in proc_lexs:
                    text_to_write.append(lex)
                    triples_to_write.append(merged_triples)

            with open(text_file, 'w') as f:
                f.writelines(s + '\n' for s in text_to_write)

            with open(graph_file, 'w') as f:
                f.writelines(s + '\n' for s in triples_to_write)

            if split == 'train':
                edge_classes_file = os.path.join(self.output_path, 'edge.classes')
                with open(edge_classes_file, 'w') as f:
                    f.writelines(s + '\n' for s in list(set(edge_classes_to_write)) + ['__no_edge__'])

    def prepare_data(self):
        if self.dataset == 'webnlg':
            self.prepareWebNLG()
        else:
            raise NotImplementedError(f'Unknown dataset {self.dataset}. Only WebNLG dataset has been implemented')

    def setup(self, stage=None):

        edge_classes_path = os.path.join(self.output_path, 'edge.classes')

        if stage == 'fit':
            text_path = os.path.join(self.output_path, 'train.text')
            graph_path = os.path.join(self.output_path, 'train.graph')
            self.dataset_train = GraphDataset(tokenizer=self.tokenizer,
                                               text_data_path=text_path,
                                               graph_data_path=graph_path,
                                               edge_classes_path=edge_classes_path,
                                               max_nodes=self.max_nodes,
                                               max_edges=self.max_edges,
                                               edges_as_classes=self.edges_as_classes)
        elif stage == 'validate':
            text_path = os.path.join(self.output_path, 'dev.text')
            graph_path = os.path.join(self.output_path, 'dev.graph')
            self.dataset_dev = GraphDataset(tokenizer=self.tokenizer,
                                            text_data_path=text_path,
                                            graph_data_path=graph_path,
                                            edge_classes_path=edge_classes_path,
                                            max_nodes=self.max_nodes,
                                            max_edges=self.max_edges,
                                            edges_as_classes=self.edges_as_classes)

        elif stage == 'test':
            text_path = os.path.join(self.output_path, 'test.text')
            graph_path = os.path.join(self.output_path, 'test.graph')
            self.dataset_test = GraphDataset(tokenizer=self.tokenizer,
                                             text_data_path=text_path,
                                             graph_data_path=graph_path,
                                             edge_classes_path=edge_classes_path,
                                             max_nodes=self.max_nodes,
                                             max_edges=self.max_edges,
                                             edges_as_classes=self.edges_as_classes)
        else:
            return

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.batch_size,
                          collate_fn=self.dataset_train._collate_fn,
                          num_workers=self.num_data_workers,
                          shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_dev,
                          batch_size=self.batch_size,
                          collate_fn=self.dataset_dev._collate_fn,
                          num_workers=self.num_data_workers,
                          shuffle=False, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                          batch_size=self.batch_size,
                          collate_fn=self.dataset_test._collate_fn,
                          num_workers=self.num_data_workers,
                          shuffle=False, pin_memory=True)


class GraphDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        text_data_path,
        graph_data_path,
        edge_classes_path,
        max_nodes,
        max_edges,
        edges_as_classes
    ):

        self.tokenizer = tokenizer
        self.max_nodes = max_nodes
        self.max_edges = max_edges

        with open(text_data_path) as f:
            self.text = f.read().splitlines()
        with open(graph_data_path) as f:
            self.graph = f.read().splitlines()
        with open(edge_classes_path) as f:
            self.edge_classes = f.read().splitlines()

        self.edges_as_classes = edges_as_classes

        self.parse_graph_data()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        item = (self.text[index], self.node[index], self.edge[index], self.edge_ind[index])
        return item

    def parse_graph_data(self):

        all_nodes = []
        all_edges = []
        all_edges_pad = []
        all_edges_ind = []

        for g_ind, g in enumerate(self.graph):
            G = nx.DiGraph()
            g += ' '
            for p in g.split('__subject__')[1:]:
                head = p.split('__predicate__')[0][1:-1]
                relop = p.split('__predicate__')[1].split('__object__')[0][1:-1]
                tail = p.split('__predicate__')[1].split('__object__')[1][1:-1]
                G.add_edge(head, tail, edge=relop)
                G.nodes[head]['node'] = head
                G.nodes[tail]['node'] = tail

            nodes = list(G.nodes) + max(0, self.max_nodes-len(G.nodes)) * ['__no_node__']

            edges = []
            edges_ind = []
            for u, v, d in G.edges(data=True):
                edges.append(d['edge'])
                edges_ind.append((nodes.index(u), nodes.index(v)))

            edges_pad = edges + max(0, self.max_edges - len(edges)) * ['__no_edge__']

            all_nodes.append(nodes)
            all_edges.append(edges)
            all_edges_pad.append(edges_pad)
            all_edges_ind.append(edges_ind)

        self.node = all_nodes
        self.edge = all_edges
        self.edge_pad = all_edges_pad
        self.edge_ind = all_edges_ind

    def _build_inputs_with_special_tokens(self, token_ids_0, _):
        # T5:   <pad_id> token_ids_0 <eos_id>
        return [self.tokenizer.pad_token_id] + token_ids_0 + [self.tokenizer.eos_token_id]

    def _collate_fn(self, data):
        text_list = []
        node_list = []
        edge_list = []
        edge_ind_list = []

        for item in data:
            text, node, edge, edge_ind = item
            text_list.append(text)
            node_list.append(node)
            edge_list.append(edge)
            edge_ind_list.append(edge_ind)

        self.tokenizer.build_inputs_with_special_tokens = self._build_inputs_with_special_tokens

        # ----------------- TEXT ----------------------

        text_batch = self.tokenizer(
            text_list,
            add_special_tokens=True,
            padding=True,
            return_tensors='pt'
        )

        collated_data = (
            text_batch['input_ids'],
            text_batch['attention_mask'],
        )

        # ------------------ NODES -------------------------
        node_list_text = []
        for node in node_list:
            node_list_text.append(' __node_sep__ '.join(node) + ' __node_sep__')

        node_batch = self.tokenizer(node_list_text,
                                    add_special_tokens=True,
                                    padding=True,
                                    return_tensors='pt'
                                    )

        collated_data += (
            node_batch['input_ids'],
            node_batch['attention_mask'],
        )

        # ----------------- EDGES ----------------------
        if self.edges_as_classes:

            edge_list_classes = [[self.edge_classes.index(edge) if edge in self.edge_classes else len(self.edge_classes)-1 for edge in edges] for edges in edge_list]

            # num_nodes X num_nodes X batch_size
            edge_mat = torch.ones(self.max_nodes, self.max_nodes, len(data))*self.edge_classes.index('__no_edge__')

            for i, (edges_cl, edge_ind) in enumerate(zip(edge_list_classes, edge_ind_list)):
                for e_p, e_i in zip(edges_cl, edge_ind):
                    edge_mat[e_i[0], e_i[1], i] = e_p

            collated_data += (edge_mat.long(),)

        else:  # edges full
            flat_edge = [node for nodes in edge_list for node in nodes]

            flat_edge_tok = self.tokenizer(
                flat_edge,
                add_special_tokens=True,
                padding=False,
                return_attention_mask=False
            )['input_ids']

            no_edge_tok = self.tokenizer(
                ['__no_edge__'],
                add_special_tokens=True,
                padding=False,
                return_attention_mask=False
            )['input_ids'][0]

            cumsum = np.cumsum([0] + [len(node) for node in edge_list])

            edge_batch = [flat_edge_tok[i:j] for (i,j) in zip(cumsum[:-1], cumsum[1:])]

            max_len = max([len(item) for item in flat_edge_tok])

            edge_batch_padded = []
            for edges in edge_batch:
                edge_batch_padded.append([i + [self.tokenizer.pad_token_id] * (max_len - len(i)) for i in edges])

            no_edge_tok = np.array(no_edge_tok + [self.tokenizer.pad_token_id] * (max_len - len(no_edge_tok)))

            edge_mat = np.tile(no_edge_tok, (len(data), self.max_nodes, self.max_nodes, 1))

            for i, (edges_padded, edge_ind) in enumerate(zip(edge_batch_padded, edge_ind_list)):
                for e_p, e_i in zip(edges_padded, edge_ind):
                    edge_mat[i, e_i[0], e_i[1]] = e_p

            edge_mat = torch.as_tensor(edge_mat).permute(1, 2, 0, 3)

            collated_data += (edge_mat,)

        return collated_data
