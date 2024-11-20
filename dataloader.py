import dgl
import torch
import pickle
import numpy as np
import os.path as osp
import dgl.function as dfn


def load_data(dataset='TMDB', return_mp=False):
    if dataset == 'TMDB':
        with open('./data/tmdb.pkl', 'rb') as f:
            network = pickle.load(f)
        num_movies = len(network['movie_labels'])
        movie_actor_mid = torch.from_numpy(network['movie-actor'][0])
        movie_actor_aid = torch.from_numpy(network['movie-actor'][1])
        movie_director_mid = torch.from_numpy(network['movie-director'][0])
        movie_director_did = torch.from_numpy(network['movie-director'][1])

        edge_dict = {
            ('movie', 'self-loop', 'movie'): (torch.arange(0, num_movies), torch.arange(0, num_movies)),
            ('actor', 'performs', 'movie'): (movie_actor_aid, movie_actor_mid),
            ('movie', 'performed_by', 'actor'): (movie_actor_mid, movie_actor_aid),
            ('director', 'directs', 'movie'): (movie_director_did, movie_director_mid),
            ('movie', 'directed_by', 'director'): (movie_director_mid, movie_director_did)
        }
        g = dgl.heterograph(edge_dict, idtype=torch.int64)
        g = dgl.to_simple(g, copy_ndata=False, writeback_mapping=False)

        # load movie features
        g.nodes['movie'].data['feat'] = torch.from_numpy(network['movie_feats'])

        # generate actor/director features
        def generate_node_features(g):
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='performed_by')
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='directed_by')
            return g

        labels = torch.tensor(network['movie_labels'], dtype=torch.int64)
        g.nodes['movie'].data['label'] = labels
        g.category = 'movie'

        movie_years = network['movie_years']
        idx_train = torch.from_numpy((movie_years <= 2015).nonzero()[0])
        idx_val = torch.from_numpy(((movie_years >= 2016) & (movie_years <= 2018)).nonzero()[0])
        idx_test = torch.from_numpy((movie_years >= 2019).nonzero()[0])
        metapaths = [['directed_by', 'directs'], ['performed_by', 'performs']]
    elif dataset == 'CroVal':
        with open('./data/croval.pkl', 'rb') as f:
            network = pickle.load(f)
        question_user_qid = torch.tensor(network['question-user'][0], dtype=torch.int32)
        question_user_uid = torch.tensor(network['question-user'][1], dtype=torch.int32)
        question_src_id = torch.tensor(network['question-question'][0], dtype=torch.int32)
        question_dst_id = torch.tensor(network['question-question'][1], dtype=torch.int32)
        question_tag_qid = torch.tensor(network['question-tag'][0], dtype=torch.int32)
        question_tag_tid = torch.tensor(network['question-tag'][1], dtype=torch.int32)

        edge_dict = {
            ('user', 'asks', 'question'): (question_user_uid, question_user_qid),
            ('question', 'asked_by', 'user'): (question_user_qid, question_user_uid),
            ('question', 'links', 'question'): (torch.cat([question_src_id, question_dst_id]),
                                                torch.cat([question_dst_id, question_src_id])),
            ('question', 'contains', 'tag'): (question_tag_qid, question_tag_tid),
            ('tag', 'contained_by', 'question'): (question_tag_tid, question_tag_qid),
        }
        g = dgl.heterograph(edge_dict, idtype=torch.int64)
        g = dgl.to_simple(g, copy_ndata=False, writeback_mapping=False)
        g = dgl.remove_self_loop(g, etype='links')
        g = dgl.add_self_loop(g, etype='links')

        # load question features
        g.nodes['question'].data['feat'] = torch.from_numpy(network['question_feats'])

        # generate user features
        def generate_node_features(g):
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='asked_by')
            return g

        # g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='contains')
        g.nodes['tag'].data['feat'] = torch.from_numpy(network['tag_feats'])

        labels = torch.tensor(network['question_labels'], dtype=torch.int64)
        g.nodes['question'].data['label'] = labels
        g.category = 'question'

        movie_years = network['question_years']
        idx_train = torch.from_numpy((movie_years <= 2015).nonzero()[0])
        idx_val = torch.from_numpy(((movie_years >= 2016) & (movie_years <= 2018)).nonzero()[0])
        idx_test = torch.from_numpy((movie_years >= 2019).nonzero()[0])
        metapaths = [['contains', 'contained_by'], ['asked_by', 'asks']]
    elif dataset == 'ArXiv':
        with open('./data/arxiv.pkl', 'rb') as f:
            network = pickle.load(f)
        paper_src_id = torch.from_numpy(network['paper-paper'][0])
        paper_dst_id = torch.from_numpy(network['paper-paper'][1])
        paper_author_pid = torch.from_numpy(network['paper-author'][0])
        paper_author_aid = torch.from_numpy(network['paper-author'][1])
        edge_dict = {
            ('paper', 'cites', 'paper'): (torch.cat([paper_src_id, paper_dst_id]),
                                          torch.cat([paper_dst_id, paper_src_id])),
            ('author', 'writes', 'paper'): (paper_author_aid, paper_author_pid),
            ('paper', 'written_by', 'author'): (paper_author_pid, paper_author_aid)
        }
        g = dgl.heterograph(edge_dict, idtype=torch.int64)
        g = dgl.to_simple(g, copy_ndata=False, writeback_mapping=False)
        g = dgl.remove_self_loop(g, etype='cites')
        g = dgl.add_self_loop(g, etype='cites')

        # load paper features
        g.nodes['paper'].data['feat'] = torch.from_numpy(network['paper_feats'])

        # generate author features
        def generate_node_features(g):
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='written_by')
            return g

        labels = torch.tensor(network['paper_labels'], dtype=torch.int64)
        g.nodes['paper'].data['label'] = labels
        g.category = 'paper'

        paper_years = network['paper_years']
        idx_train = torch.from_numpy((paper_years <= 2017).nonzero()[0])
        idx_val = torch.from_numpy(((paper_years >= 2018) & (paper_years <= 2018)).nonzero()[0])
        idx_test = torch.from_numpy((paper_years >= 2019).nonzero()[0])
        metapaths = [['cites'], ['written_by', 'writes']]
    elif 'IGB' in dataset:
        dataset_size = 'tiny' if dataset.split('-')[-2] == '549K' else 'small'
        num_classes = 19 if dataset.split('-')[-1] == '19' else 2983
        dir_path = osp.join('data', 'igb', dataset_size, 'processed')

        paper_paper_edges = torch.from_numpy(
            np.load(osp.join(dir_path, 'paper__cites__paper', 'edge_index.npy')))
        paper_author_edges = torch.from_numpy(
            np.load(osp.join(dir_path, 'paper__written_by__author', 'edge_index.npy')))
        author_institute_edges = torch.from_numpy(
            np.load(osp.join(dir_path, 'author__affiliated_to__institute', 'edge_index.npy')))
        paper_fos_edges = torch.from_numpy(
            np.load(osp.join(dir_path, 'paper__topic__fos', 'edge_index.npy')))

        edge_dict = {
            # ('paper', 'cites', 'paper'): (torch.cat([paper_paper_edges[:, 0], paper_paper_edges[:, 1]]),
            #                               torch.cat([paper_paper_edges[:, 1], paper_paper_edges[:, 0]])),
            ('paper', 'cites', 'paper'): (paper_paper_edges[:, 0], paper_paper_edges[:, 1]),
            ('author', 'writes', 'paper'): (paper_author_edges[:, 1], paper_author_edges[:, 0]),
            ('paper', 'written_by', 'author'): (paper_author_edges[:, 0], paper_author_edges[:, 1]),

            ('author', 'affiliated_with', 'institute'): (
                author_institute_edges[:, 0],
                author_institute_edges[:, 1],
            ),
            ('institute', 'affiliates', 'author'): (
                author_institute_edges[:, 1], author_institute_edges[:, 0]),
            ('fos', 'topics', 'paper'): (paper_fos_edges[:, 1], paper_fos_edges[:, 0]),
            ('paper', 'has_topic', 'fos'): (paper_fos_edges[:, 0], paper_fos_edges[:, 1])
        }
        g = dgl.heterograph(edge_dict)
        g = dgl.to_simple(g, copy_ndata=False, writeback_mapping=False)
        g = dgl.remove_self_loop(g, etype='cites')
        g = dgl.add_self_loop(g, etype='cites')

        # load paper features
        paper_features = torch.from_numpy(np.load(osp.join(dir_path, 'paper', 'node_feat.npy')))
        g.nodes['paper'].data['feat'] = paper_features

        # generate author/institute features
        def generate_node_features(g):
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='written_by')
            g.update_all(dfn.copy_u('feat', 'm'), dfn.mean('m', 'feat'), etype='affiliated_with')
            return g

        # load fos features
        fos_features = torch.from_numpy(np.load(osp.join(dir_path, 'fos', 'node_feat.npy')))
        g.nodes['fos'].data['feat'] = fos_features

        if num_classes == 19:
            paper_labels = torch.tensor(
                np.load(osp.join(dir_path, 'paper', 'node_label_19.npy')), dtype=torch.int64)
        else:
            paper_labels = torch.tensor(
                np.load(osp.join(dir_path, 'paper', 'node_label_2K.npy')), dtype=torch.int64)
        g.nodes['paper'].data['label'] = paper_labels
        g.category = 'paper'

        paper_years = np.load(osp.join(dir_path, 'paper', 'paper_year.npy'))
        idx_train = torch.from_numpy((paper_years <= 2016).nonzero()[0])
        idx_val = torch.from_numpy(((paper_years >= 2017) & (paper_years <= 2018)).nonzero()[0])
        idx_test = torch.from_numpy((paper_years >= 2019).nonzero()[0])
        metapaths = [['cites'], ['written_by', 'writes']]

    if return_mp:
        return g, (idx_train, idx_val, idx_test), generate_node_features, metapaths

    return g, (idx_train, idx_val, idx_test), generate_node_features
