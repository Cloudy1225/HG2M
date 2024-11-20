import copy
import json
import time, datetime
from tqdm import tqdm

import dgl
import torch
import numpy as np

import torch.optim as optim
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

from models import get_model

"""
1. Train and eval
"""


def train_gnn(model, category, dataloader, optimizer, loss_fcn=F.cross_entropy):
    model.train()
    device = next(model.parameters()).device
    total_loss = 0

    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [block.to(device) for block in blocks]
        x = blocks[0].srcdata['feat']
        y_true = blocks[-1].dstdata['label'][category]
        logits = model(blocks, x)
        loss = loss_fcn(logits, y_true)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


def eval_gnn(model, category, dataloader, evaluator, return_log_prob=False):
    model.eval()
    device = next(model.parameters()).device
    y_true = []
    logits = []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [block.to(device) for block in blocks]
            x = blocks[0].srcdata['feat']
            y_true.append(blocks[-1].dstdata['label'][category])
            logits.append(model(blocks, x))

        y_true = torch.cat(y_true)
        logits = torch.cat(logits)
        acc = evaluator(logits, y_true)

    if return_log_prob:
        return acc, logits.log_softmax(dim=1)

    return acc


def train_mlp(model, feats, labels, batch_size, optimizer, loss_fcn=F.nll_loss, lamda=1.):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    lamb: weight parameter lambda
    """
    model.train()
    device = next(model.parameters()).device
    feats = feats.to(device)
    labels = labels.to(device)

    logits = model(None, feats)
    y_true = labels
    loss = loss_fcn(logits.log_softmax(dim=1), y_true)
    loss *= lamda

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def eval_mlp(model, feats, labels, evaluator):
    model.eval()
    device = next(model.parameters()).device
    feats = feats.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        logits = model(None, feats)
        acc = evaluator(logits, labels)

    return acc


"""
2. Run teacher
"""


def evaluator(logits, y_true):
    y_pred = logits.argmax(1)
    return float(y_pred.eq(y_true).float().mean()) * 100


def run_transductive(
        conf,
        g: dgl.heterograph,
        splits,
        log=print
):
    device = conf['device']
    category = conf['category']
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    model_type = conf['model_type']

    idx_train, idx_val, idx_test = splits

    if 'MLP' not in model_type:
        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves momory and CPU.
        g.create_formats_()
        sampler = dgl.dataloading.NeighborSampler(eval(str(conf['fan_out'])),
                                                  prefetch_node_feats={k: ['feat'] for k in g.ntypes},
                                                  prefetch_labels={category: ['label']})
        train_dataloader = dgl.dataloading.DataLoader(
            g, {category: idx_train}, sampler,
            batch_size=batch_size,
            shuffle=True, drop_last=False,
            num_workers=num_workers)

        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(conf['num_layers'])

        val_dataloader = dgl.dataloading.DataLoader(
            g, {category: idx_val}, sampler_eval,
            batch_size=batch_size,
            shuffle=False, drop_last=False,
            num_workers=num_workers)

        test_dataloader = dgl.dataloading.DataLoader(
            g, {category: idx_test}, sampler_eval,
            batch_size=batch_size,
            shuffle=False, drop_last=False,
            num_workers=num_workers)
    else:
        feats = g.ndata['feat'][category]
        labels = g.ndata['label'][category]
        feats_train, labels_train = feats[idx_train], labels[idx_train]
        feats_val, labels_val = feats[idx_val], labels[idx_val]
        feats_test, labels_test = feats[idx_test], labels[idx_test]

    in_dim = g.ndata['feat'][category].shape[1]
    num_classes = max(g.ndata['label'][category]) + 1

    model = get_model(model_type, g.etypes, in_dim, num_classes, conf).to(device)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    log(f'model size: {size_all_mb:.3f}MB')

    optimizer = optim.Adam(model.parameters(),
                           lr=conf['learning_rate'], weight_decay=conf['weight_decay'])

    best_epoch = 0
    best_acc_val = 0
    training_start = time.time()
    for epoch in range(1, conf['max_epoch'] + 1):
        epoch_start = time.time()
        if 'MLP' not in model_type:
            loss = train_gnn(model, category, train_dataloader, optimizer, F.cross_entropy)
        else:
            loss = train_mlp(model, feats_train, labels_train, batch_size, optimizer, F.nll_loss)

        if epoch % conf['eval_interval'] == 0:
            if 'MLP' not in model_type:
                acc_val = eval_gnn(model, category, val_dataloader, evaluator)
            else:
                acc_val = eval_mlp(model, feats_val, labels_val, evaluator)
            if best_acc_val < acc_val:
                best_epoch = epoch
                best_acc_val = acc_val
                if conf['model_save']:
                    torch.save(model.state_dict(), conf['teacher_path'])
                else:
                    best_state = copy.deepcopy(model.state_dict())
            time_taken = str(datetime.timedelta(seconds=(time.time() - epoch_start)))
            log(
                f'Epoch {epoch:03d} | Loss={loss:.4f} | '
                f'Val={acc_val:.2f} | Time {time_taken}'
            )

    if conf['model_save']:
        model.load_state_dict(torch.load(conf['teacher_path']))
    else:
        model.load_state_dict(best_state)
    if 'MLP' not in model_type:
        acc_test = eval_gnn(model, category, test_dataloader, evaluator)
    else:
        acc_test = eval_mlp(model, feats_test, labels_test, evaluator)
    time_taken = str(datetime.timedelta(seconds=(time.time() - training_start)))
    log(f'Best Epoch {best_epoch:3d} | Test={acc_test :.2f} | Total time taken {time_taken}')

    return acc_test


def run_inductive(
        conf,
        obs_g: dgl.heterograph,
        g: dgl.heterograph,
        splits,
        log=print
):
    device = conf['device']
    category = conf['category']
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    model_type = conf['model_type']

    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = splits

    if 'MLP' not in model_type:
        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves memory and CPU.
        g.create_formats_()
        sampler = dgl.dataloading.NeighborSampler(eval(str(conf['fan_out'])),
                                                  prefetch_node_feats={k: ['feat'] for k in g.ntypes},
                                                  prefetch_labels={category: ['label']})
        train_dataloader = dgl.dataloading.DataLoader(
            obs_g, {category: obs_idx_train}, sampler,
            batch_size=batch_size,
            shuffle=True, drop_last=False,
            num_workers=num_workers)

        sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(conf['num_layers'])
        val_dataloader = dgl.dataloading.DataLoader(
            obs_g, {category: obs_idx_val}, sampler_eval,
            batch_size=batch_size,
            shuffle=False, drop_last=False,
            num_workers=num_workers)
        test_tran_dataloader = dgl.dataloading.DataLoader(
            obs_g, {category: obs_idx_test}, sampler_eval,
            batch_size=batch_size,
            shuffle=False, drop_last=False,
            num_workers=num_workers)
        test_ind_dataloader = dgl.dataloading.DataLoader(
            g, {category: idx_test_ind}, sampler_eval,
            batch_size=batch_size,
            shuffle=False, drop_last=False,
            num_workers=num_workers)
    else:
        feats = g.ndata['feat'][category]
        labels = g.ndata['label'][category]
        feats_train, labels_train = feats[idx_obs][obs_idx_train], labels[idx_obs][obs_idx_train]
        feats_val, labels_val = feats[idx_obs][obs_idx_val], labels[idx_obs][obs_idx_val]
        feats_test_tran, labels_test_tran = feats[idx_obs][obs_idx_test], labels[idx_obs][obs_idx_test]
        feats_test_ind, labels_test_ind = feats[idx_test_ind], labels[idx_test_ind]

    in_dim = obs_g.ndata['feat'][category].shape[1]
    num_classes = max(obs_g.ndata['label'][category]) + 1

    model = get_model(model_type, obs_g.etypes, in_dim, num_classes, conf).to(device)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    log(f'model size: {size_all_mb:.3f}MB')

    optimizer = optim.Adam(model.parameters(),
                           lr=conf['learning_rate'], weight_decay=conf['weight_decay'])

    best_epoch = 0
    best_acc_val = 0
    training_start = time.time()
    for epoch in range(1, conf['max_epoch'] + 1):
        epoch_start = time.time()
        if 'MLP' not in model_type:
            loss = train_gnn(model, category, train_dataloader, optimizer, F.cross_entropy)
        else:
            loss = train_mlp(model, feats_train, labels_train, batch_size, optimizer, F.nll_loss)

        if epoch % conf['eval_interval'] == 0:
            if 'MLP' not in model_type:
                acc_val = eval_gnn(model, category, val_dataloader, evaluator)
            else:
                acc_val = eval_mlp(model, feats_val, labels_val, evaluator)
            if best_acc_val < acc_val:
                best_epoch = epoch
                best_acc_val = acc_val
                if conf['model_save']:
                    torch.save(model.state_dict(), conf['teacher_path'])
                else:
                    best_state = copy.deepcopy(model.state_dict())
            time_taken = str(datetime.timedelta(seconds=(time.time() - epoch_start)))
            log(
                f'Epoch {epoch:03d} | Loss={loss:.4f} | '
                f'Val={acc_val:.2f} | Time {time_taken}'
            )

    if conf['model_save']:
        model.load_state_dict(torch.load(conf['teacher_path']))
    else:
        model.load_state_dict(best_state)

    if 'MLP' not in model_type:
        acc_test_tran = eval_gnn(model, category, test_tran_dataloader, evaluator)
        acc_test_ind = eval_gnn(model, category, test_ind_dataloader, evaluator)
    else:
        acc_test_tran = eval_mlp(model, feats_test_tran, labels_test_tran, evaluator)
        acc_test_ind = eval_mlp(model, feats_test_ind, labels_test_ind, evaluator)
    acc_test = (1 - conf['split_rate']) * acc_test_tran + conf['split_rate'] * acc_test_ind

    time_taken = str(datetime.timedelta(seconds=(time.time() - training_start)))
    log(f'Best Epoch {best_epoch:3d} | Test={acc_test :.2f} '
        f'Tran={acc_test_tran :.2f} Ind={acc_test_ind :.2f}| Total time taken {time_taken}')

    return acc_test, acc_test_tran, acc_test_ind


"""
3. Distill
"""


def get_logits(model_type, g, category, conf, batch_size, num_workers, device):
    # get teacher's outputs
    g.create_formats_()
    teacher_conf = json.load(fp=open(conf['teacher_conf_path'], 'r'))
    sampler_eval = dgl.dataloading.MultiLayerFullNeighborSampler(teacher_conf['num_layers'])
    eval_dataloader = dgl.dataloading.DataLoader(
        g, {category: torch.arange(g.num_nodes(category))}, sampler_eval,
        batch_size=50000,
        shuffle=False, drop_last=False,
        num_workers=num_workers)

    in_dim = g.ndata['feat'][category].shape[1]
    num_classes = max(g.ndata['label'][category]) + 1

    model = get_model(model_type, g.etypes, in_dim, num_classes, teacher_conf).to(device)
    model.load_state_dict(torch.load(conf['teacher_path']))

    model.eval()
    with torch.no_grad():
        if g.num_nodes(category) >= 500000:
            logits = []
            for input_nodes, output_nodes, blocks in tqdm(eval_dataloader):
                blocks = [block.to(device) for block in blocks]
                # x = blocks[0].srcdata['feat']
                logits.append(model(blocks, blocks[0].srcdata['feat']))
            logits = torch.cat(logits)
        else:
            g = g.to(device)
            logits = model([g] * len(model.layers), g.ndata['feat'])

    return logits


def get_reliable_node_indices(probs, labels, p, idx_train, idx_val, idx_test):
    # correctly predicted nodes
    true_train_mask = (probs[idx_train].argmax(1) == labels[idx_train]).cpu()
    reliable_train_idx = idx_train[true_train_mask]

    # low entropy and high confidence
    idx_unlabeled = torch.cat((idx_val, idx_test))
    probs_unlabeled = probs[idx_unlabeled]
    entropy = -torch.sum(probs_unlabeled * torch.log(probs_unlabeled), dim=1)
    confidence = probs_unlabeled.max(dim=1)[0]
    k = int(p * len(probs_unlabeled))
    entropy_threshold = torch.topk(entropy, k=k, largest=False)[0][-1]
    confidence_threshold = torch.topk(confidence, k=k)[0][-1]
    mask = ((entropy <= entropy_threshold) & (confidence >= confidence_threshold)).cpu()
    reliable_unlabeled_idx = idx_unlabeled[mask]

    reliable_idx = torch.cat((reliable_train_idx, reliable_unlabeled_idx))
    acc = 100 * (probs[reliable_idx].argmax(1) == labels[reliable_idx]).sum() / reliable_idx.shape[0]
    print(f'{reliable_idx.shape[0]} reliable nodes ({reliable_idx.shape[0] / labels.shape[0] * 100:.2f}%), '
          f'{reliable_train_idx.shape[0]} training nodes, '
          f'{reliable_unlabeled_idx.shape[0]} unlabeled nodes, '
          f'ACC {acc:.2f}')
    return reliable_idx


def get_candidate_edge_indices(g, metapath, reliable_node_idx, min_n=1, labels=None):
    adj = 1
    for etype in metapath:
        adj = adj * g.adj_external(
            etype=etype, scipy_fmt="csr", transpose=False
        )
    adj = adj.tocoo()
    mask = adj.row < adj.col
    src = adj.row[mask]
    dst = adj.col[mask]
    num_cons = adj.data[mask]
    mask = num_cons >= min_n
    src = src[mask]
    dst = dst[mask]
    num_cons = num_cons[mask]

    def delete_edges_without_reliable_node(src, dst, num_cons, reliable_node_idx, num_total_nodes):
        unreliable_mask = np.full((num_total_nodes,), True, dtype=bool)
        unreliable_mask[reliable_node_idx] = False
        edge_with_reliable_node_mask = ~(unreliable_mask[src] & unreliable_mask[dst])
        src = src[edge_with_reliable_node_mask]
        dst = dst[edge_with_reliable_node_mask]
        num_cons = num_cons[edge_with_reliable_node_mask]
        return src, dst, num_cons

    src, dst, num_cons = delete_edges_without_reliable_node(src, dst, num_cons, reliable_node_idx, adj.shape[0])

    if labels is not None:
        homophily = (labels[src] == labels[dst]).sum() / src.shape[0]
        print(f'{src.shape[0]} candidate edges with homophily {homophily:.4f}')

    return src, dst, num_cons


def get_reliable_edge_indices(src, dst, num_cons, idx_train, feats, probs, labels, reliable_node_idx):
    train_mask = np.full((labels.shape[0],), False, dtype=bool)
    train_mask[idx_train] = True
    train_edge_mask = train_mask[src] & train_mask[dst]
    test_edge_mask = ~train_edge_mask

    def compute_x1(feats, src, dst, batch_size=500000):
        x1_list = []

        for i in range(0, len(src), batch_size):
            src_batch = src[i:i + batch_size]
            dst_batch = dst[i:i + batch_size]

            x1_batch = F.cosine_similarity(feats[src_batch], feats[dst_batch], dim=1).cpu().numpy()
            x1_list.append(x1_batch)

        return np.concatenate(x1_list)

    def compute_x3(probs, src, dst, batch_size=500000):
        x3_list = []

        for i in range(0, len(src), batch_size):
            src_batch = src[i:i + batch_size]
            dst_batch = dst[i:i + batch_size]

            x3_batch = (probs[src_batch] * probs[dst_batch]).sum(1).cpu().numpy()
            x3_list.append(x3_batch)

        return np.concatenate(x3_list)

    x1 = compute_x1(feats, src, dst)
    x2 = num_cons
    x3 = compute_x3(probs, src, dst)

    X = np.stack((x1, x2, x3), axis=1)
    y = (labels[src] == labels[dst]).cpu().numpy()

    X_train = X[train_edge_mask]
    y_train = y[train_edge_mask]
    X_test = X[test_edge_mask]
    y_test = y[test_edge_mask]

    print(f'{y_train.shape[0]} train edges and {y_test.shape[0]} test edges')

    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'LogisticRegression {clf.coef_[0]}')

    print(
        f'ACC {accuracy_score(y_test, y_pred):.4f} AUC {roc_auc_score(y_test, y_pred):.4f}, '
        f'Precision {precision_score(y_test, y_pred):.4f}, Recall {recall_score(y_test, y_pred):.4f}')

    reliable_src = np.concatenate([src[train_edge_mask][y_train], src[test_edge_mask][y_pred]])
    reliable_dst = np.concatenate([dst[train_edge_mask][y_train], dst[test_edge_mask][y_pred]])
    reliable_score = np.concatenate([np.ones((y_train.sum(),)), clf.predict_proba(X_test[y_pred])[:, 1]])

    # to directed
    reliable_mask = np.full((labels.shape[0],), False, dtype=bool)
    reliable_mask[reliable_node_idx] = True

    src2dst = reliable_mask[reliable_src]
    dst2src = reliable_mask[reliable_dst]

    src = np.concatenate([reliable_src[src2dst], reliable_dst[dst2src]])
    dst = np.concatenate([reliable_dst[src2dst], reliable_src[dst2src]])
    score = np.concatenate([reliable_score[src2dst], reliable_score[dst2src]])

    score = (score - score.min()) / (score.max() - score.min())
    mask = score > 0.01
    src = src[mask]
    dst = dst[mask]
    score = score[mask]

    homophily = (labels[src] == labels[dst]).sum() / src.shape[0]
    print(f'{src.shape[0]} reliable edges with homophily {homophily:.4f}')

    return src, dst, score


def reliable_edge_indices(g, feats, probs, labels, idx_train, idx_val, idx_test, metapath, p=0.8, min_n=1,
                          return_nodes=False):
    print('metapath', metapath)
    reliable_node_idx = get_reliable_node_indices(probs, labels, p, idx_train, idx_val, idx_test)
    src, dst, num_cons = get_candidate_edge_indices(g, metapath, reliable_node_idx, min_n,
                                                    labels=labels)
    reliable_src, reliable_dst, reliable_score = get_reliable_edge_indices(src, dst, num_cons, idx_train, feats, probs,
                                                                           labels, reliable_node_idx)

    if return_nodes:
        return reliable_src, reliable_dst, reliable_score, reliable_node_idx
    return reliable_src, reliable_dst, reliable_score


def distill_train_mlp(model, feats, log_target, src, dst, weight, optimizer, lamda=1.):
    """
    Train MLP for large datasets. Process the data in mini-batches. The graph is ignored, node features only.
    lamb: weight parameter lambda
    """
    model.train()
    device = next(model.parameters()).device
    feats = feats.to(device)
    log_target = log_target.to(device)

    # src = np.concatenate([np.arange(feats.shape[0]), src])
    # dst = np.concatenate([np.arange(feats.shape[0]), dst])
    # weight = torch.concatenate([torch.ones((feats.shape[0],), device=device), weight])

    # _, embeds = model(None, feats, return_embeddings=True)
    # embeds = F.normalize(embeds, p=2, dim=1)
    # loss = -(embeds[src] * embeds[dst]).sum() / src.shape[0]

    logits = model(None, feats)
    log_probs = F.log_softmax(logits, dim=1)
    kl = F.kl_div(log_probs[dst], log_target[src], reduction='none', log_target=True)
    loss = ((kl.sum(dim=1) * weight) / src.shape[0]).sum()
    loss *= lamda

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def distill_run_transductive(
        conf,
        g: dgl.heterograph,
        splits,
        log=print
):
    """
    Distill training and eval under the transductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively
    loss_and_score: Stores losses and scores.
    """
    device = conf['device']
    category = conf['category']
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    model_type = conf['teacher']
    lamda = conf['lamda']

    if model_type in ['SimpleHGN', 'ieHGCN']:
        logits = torch.load(f"{conf['output_dir']}/out.pt").detach().to(device)
    else:
        logits = get_logits(model_type, g, category, conf, batch_size, num_workers, device)

    idx_train, idx_val, idx_test = splits

    feats = g.ndata['feat'][category].to(device)
    labels = g.ndata['label'][category].to(device)
    feats_train, labels_train = feats[idx_train], labels[idx_train]
    feats_val, labels_val = feats[idx_val], labels[idx_val]
    feats_test, labels_test = feats[idx_test], labels[idx_test]

    # get teacher's outputs
    in_dim = g.ndata['feat'][category].shape[1]
    num_classes = max(g.ndata['label'][category]) + 1

    log(f'Teacher {model_type} Train ACC {evaluator(logits[idx_train], labels_train):.2f} '
        f'Val ACC {evaluator(logits[idx_val], labels_val):.2f} '
        f'Test ACC {evaluator(logits[idx_test], labels_test):.2f}')
    log_target = logits.log_softmax(dim=1)

    del g, logits

    # distill into MLP
    model = get_model('MLP', None, in_dim, num_classes, conf).to(device)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    log(f'model size: {size_all_mb:.3f}MB')

    # loss_fcn1 = torch.nn.NLLLoss()
    # loss_fcn2 = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    optimizer = optim.Adam(model.parameters(),
                           lr=conf['learning_rate'], weight_decay=conf['weight_decay'])

    best_epoch = 0
    best_acc_val = 0
    training_start = time.time()

    for epoch in range(1, conf['max_epoch'] + 1):
        epoch_start = time.time()

        model.train()

        log_probs = F.log_softmax(model(None, feats), dim=1)
        loss_ce = F.nll_loss(log_probs[idx_train], labels[idx_train])

        loss_kl = F.kl_div(log_probs, log_target, reduction='batchmean', log_target=True)

        loss = lamda * loss_ce + (1 - lamda) * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'CE={loss_ce:.4f} KL={loss_kl:.4f}')

        if epoch % conf['eval_interval'] == 0:
            acc_train = eval_mlp(model, feats_train, labels_train, evaluator)
            acc_val = eval_mlp(model, feats_val, labels_val, evaluator)
            acc_test = eval_mlp(model, feats_test, labels_test, evaluator)
            if best_acc_val < acc_val:
                best_epoch = epoch
                best_acc_val = acc_val
                if conf['model_save']:
                    torch.save(model.state_dict(), conf['student_path'])
                else:
                    best_state = copy.deepcopy(model.state_dict())
            time_taken = str(datetime.timedelta(seconds=(time.time() - epoch_start)))
            log(
                f'Epoch {epoch:03d} | Loss={loss:.4f} | Train={acc_train:.2f} | '
                f'Val={acc_val:.2f} Test={acc_test:.2f} | Time {time_taken}'
            )

    if conf['model_save']:
        model.load_state_dict(torch.load(conf['student_path']))
    else:
        model.load_state_dict(best_state)

    acc_test = eval_mlp(model, feats_test, labels_test, evaluator)
    time_taken = str(datetime.timedelta(seconds=(time.time() - training_start)))
    log(f'Best Epoch {best_epoch:3d} | Test={acc_test :.2f} | Total time taken {time_taken}')

    return acc_test


def distill_run_transductive_plus(
        conf,
        g: dgl.heterograph,
        splits,
        log=print
):
    """
    Distill training and eval under the transductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively
    loss_and_score: Stores losses and scores.
    """
    device = conf['device']
    category = conf['category']
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    model_type = conf['teacher']
    lamda = conf['lamda']
    gamma = conf['gamma']

    if model_type in ['SimpleHGN', 'ieHGCN']:
        logits = torch.load(f"{conf['output_dir']}/out.pt").detach().to(device)
    else:
        logits = get_logits(model_type, g, category, conf, batch_size, num_workers, device)

    idx_train, idx_val, idx_test = splits

    feats = g.ndata['feat'][category].to(device)
    labels = g.ndata['label'][category].to(device)
    feats_train, labels_train = feats[idx_train], labels[idx_train]
    feats_val, labels_val = feats[idx_val], labels[idx_val]
    feats_test, labels_test = feats[idx_test], labels[idx_test]

    # get teacher's outputs
    in_dim = g.ndata['feat'][category].shape[1]
    num_classes = max(g.ndata['label'][category]) + 1

    log(f'Teacher {model_type} Train ACC {evaluator(logits[idx_train], labels_train):.2f} '
        f'Val ACC {evaluator(logits[idx_val], labels_val):.2f} '
        f'Test ACC {evaluator(logits[idx_test], labels_test):.2f}')
    log_target = logits.log_softmax(dim=1)

    probs = F.softmax(logits, dim=1)
    metapaths = conf['metapaths']
    min_ns = conf['min_ns']
    reliable_src_list = []
    reliable_dst_list = []
    reliable_score_list = []
    reliable_node_indices = None
    for metapath, min_n in zip(metapaths, min_ns):
        reliable_src, reliable_dst, reliable_score, reliable_node_indices = reliable_edge_indices(g, feats, probs,
                                                                                                  labels,
                                                                                                  idx_train, idx_val,
                                                                                                  idx_test,
                                                                                                  metapath, p=conf['p'],
                                                                                                  min_n=min_n,
                                                                                                  return_nodes=True)
        reliable_score = torch.tensor(reliable_score, dtype=torch.float32).to(device)
        reliable_src_list.append(reliable_src)
        reliable_dst_list.append(reliable_dst)
        reliable_score_list.append(reliable_score)

    del g, logits, probs

    # distill into MLP
    model = get_model('MLP', None, in_dim, num_classes, conf).to(device)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    log(f'model size: {size_all_mb:.3f}MB')

    # loss_fcn1 = torch.nn.NLLLoss()
    # loss_fcn2 = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    optimizer = optim.Adam(model.parameters(),
                           lr=conf['learning_rate'], weight_decay=conf['weight_decay'])

    best_epoch = 0
    best_acc_val = 0
    training_start = time.time()

    for epoch in range(1, conf['max_epoch'] + 1):
        epoch_start = time.time()

        model.train()
        logits = model(None, feats)
        log_probs = F.log_softmax(logits, dim=1)
        loss_ce = F.nll_loss(log_probs[idx_train], labels[idx_train])

        # loss_kl = F.kl_div(log_probs, log_target, reduction='batchmean', log_target=True)
        loss_kl = F.kl_div(log_probs[reliable_node_indices], log_target[reliable_node_indices], reduction='batchmean',
                           log_target=True)

        loss_nd = 0.
        for reliable_src, reliable_dst, reliable_score in zip(reliable_src_list, reliable_dst_list,
                                                              reliable_score_list):
            kl = F.kl_div(log_probs[reliable_dst], log_target[reliable_src], reduction='none', log_target=True)
            kl = (kl.sum(dim=1) * reliable_score)
            loss_nd += (kl.sum() + loss_kl * log_probs.shape[0]) / (reliable_src.shape[0] + log_probs.shape[0])
        loss_nd /= len(reliable_src_list)

        # loss = lamda * loss_ce + (1 - lamda) * loss_kl
        loss = lamda * loss_ce + (1 - lamda) * (loss_kl + gamma * loss_nd)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'CE={loss_ce:.4f} KL={loss_kl:.4f} ND={loss_nd:.4f}')

        if epoch % conf['eval_interval'] == 0:
            acc_train = eval_mlp(model, feats_train, labels_train, evaluator)
            acc_val = eval_mlp(model, feats_val, labels_val, evaluator)
            acc_test = eval_mlp(model, feats_test, labels_test, evaluator)
            if best_acc_val < acc_val:
                best_epoch = epoch
                best_acc_val = acc_val
                if conf['model_save']:
                    torch.save(model.state_dict(), conf['student_path'])
                else:
                    best_state = copy.deepcopy(model.state_dict())
            time_taken = str(datetime.timedelta(seconds=(time.time() - epoch_start)))
            log(
                f'Epoch {epoch:03d} | Loss={loss:.4f} | Train={acc_train:.2f} | '
                f'Val={acc_val:.2f} Test={acc_test:.2f} | Time {time_taken}'
            )

    if conf['model_save']:
        model.load_state_dict(torch.load(conf['student_path']))
    else:
        model.load_state_dict(best_state)

    acc_test = eval_mlp(model, feats_test, labels_test, evaluator)
    time_taken = str(datetime.timedelta(seconds=(time.time() - training_start)))
    log(f'Best Epoch {best_epoch:3d} | Test={acc_test :.2f} | Total time taken {time_taken}')

    return acc_test


def distill_run_inductive(
        conf,
        obs_g: dgl.heterograph,
        g: dgl.heterograph,
        splits,
        log=print
):
    """
    Distill training and eval under the transductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively
    loss_and_score: Stores losses and scores.
    """
    device = conf['device']
    category = conf['category']
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    model_type = conf['teacher']
    lamda = conf['lamda']

    logits = get_logits(model_type, obs_g, category, conf, batch_size, num_workers, device)

    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = splits

    feats_obs = obs_g.ndata['feat'][category].to(device)
    labels_obs = obs_g.ndata['label'][category].to(device)
    feats_train, labels_train = feats_obs[obs_idx_train], labels_obs[obs_idx_train]
    feats_val, labels_val = feats_obs[obs_idx_val], labels_obs[obs_idx_val]
    feats_test_tran, labels_test_tran = feats_obs[obs_idx_test], labels_obs[obs_idx_test]
    feats_test_ind = g.ndata['feat'][category][idx_test_ind].to(device)
    labels_test_ind = g.ndata['label'][category][idx_test_ind].to(device)

    # get teacher's outputs
    in_dim = feats_train.shape[1]
    num_classes = max(labels_train) + 1

    log(f'Teacher {model_type} Train ACC {evaluator(logits[obs_idx_train], labels_train):.2f} '
        f'Val ACC {evaluator(logits[obs_idx_val], labels_val):.2f} '
        f'Test_tran ACC {evaluator(logits[obs_idx_test], labels_test_tran):.2f}')
    log_target = logits.log_softmax(dim=1)

    del obs_g, g, logits

    # distill into MLP
    model = get_model('MLP', None, in_dim, num_classes, conf).to(device)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    log(f'model size: {size_all_mb:.3f}MB')

    # loss_fcn1 = torch.nn.NLLLoss()
    # loss_fcn2 = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    optimizer = optim.Adam(model.parameters(),
                           lr=conf['learning_rate'], weight_decay=conf['weight_decay'])

    best_epoch = 0
    best_acc_val = 0
    training_start = time.time()

    for epoch in range(1, conf['max_epoch'] + 1):
        epoch_start = time.time()

        model.train()

        log_probs = F.log_softmax(model(None, feats_obs), dim=1)
        loss_ce = F.nll_loss(log_probs[obs_idx_train], labels_train)

        loss_kl = F.kl_div(log_probs, log_target, reduction='batchmean', log_target=True)

        loss = lamda * loss_ce + (1 - lamda) * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'CE={loss_ce:.4f} KL={loss_kl:.4f}')

        if epoch % conf['eval_interval'] == 0:
            acc_train = eval_mlp(model, feats_train, labels_train, evaluator)
            acc_val = eval_mlp(model, feats_val, labels_val, evaluator)
            acc_test_tran = eval_mlp(model, feats_test_tran, labels_test_tran, evaluator)
            acc_test_ind = eval_mlp(model, feats_test_ind, labels_test_ind, evaluator)
            acc_test = (1 - conf['split_rate']) * acc_test_tran + conf['split_rate'] * acc_test_ind

            if best_acc_val < acc_val:
                best_epoch = epoch
                best_acc_val = acc_val
                if conf['model_save']:
                    torch.save(model.state_dict(), conf['student_path'])
                else:
                    best_state = copy.deepcopy(model.state_dict())
            time_taken = str(datetime.timedelta(seconds=(time.time() - epoch_start)))
            log(
                f'Epoch {epoch:03d} | Loss={loss:.4f} | Train={acc_train:.2f} | Val={acc_val:.2f} '
                f'Test={acc_test:.2f} Tran={acc_test_tran :.2f} Ind={acc_test_ind :.2f} | Time {time_taken}'
            )

    if conf['model_save']:
        model.load_state_dict(torch.load(conf['student_path']))
    else:
        model.load_state_dict(best_state)

    acc_test_tran = eval_mlp(model, feats_test_tran, labels_test_tran, evaluator)
    acc_test_ind = eval_mlp(model, feats_test_ind, labels_test_ind, evaluator)
    acc_test = (1 - conf['split_rate']) * acc_test_tran + conf['split_rate'] * acc_test_ind

    time_taken = str(datetime.timedelta(seconds=(time.time() - training_start)))
    log(f'Best Epoch {best_epoch:3d} | Test={acc_test :.2f} '
        f'Tran={acc_test_tran :.2f} Ind={acc_test_ind :.2f}| Total time taken {time_taken}')

    return acc_test, acc_test_tran, acc_test_ind


def distill_run_inductive_plus(
        conf,
        obs_g: dgl.heterograph,
        g: dgl.heterograph,
        splits,
        log=print
):
    """
    Distill training and eval under the transductive setting.
    The hard_label_train/soft_label_train/valid/test split is specified by `distill_indices`.
    The input graph is assumed to be large, and MLP is assumed to be the student model. Thus, node feature only and mini-batch is used.

    out_t: Soft labels produced by the teacher model.
    criterion_l & criterion_t: Loss used for hard labels (`labels`) and soft labels (`out_t`) respectively
    loss_and_score: Stores losses and scores.
    """
    device = conf['device']
    category = conf['category']
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    model_type = conf['teacher']
    lamda = conf['lamda']
    gamma = conf['gamma']

    logits = get_logits(model_type, obs_g, category, conf, batch_size, num_workers, device)

    obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind = splits

    feats_obs = obs_g.ndata['feat'][category].to(device)
    labels_obs = obs_g.ndata['label'][category].to(device)
    feats_train, labels_train = feats_obs[obs_idx_train], labels_obs[obs_idx_train]
    feats_val, labels_val = feats_obs[obs_idx_val], labels_obs[obs_idx_val]
    feats_test_tran, labels_test_tran = feats_obs[obs_idx_test], labels_obs[obs_idx_test]
    feats_test_ind = g.ndata['feat'][category][idx_test_ind].to(device)
    labels_test_ind = g.ndata['label'][category][idx_test_ind].to(device)

    # get teacher's outputs
    in_dim = feats_train.shape[1]
    num_classes = max(labels_train) + 1

    log(f'Teacher {model_type} Train ACC {evaluator(logits[obs_idx_train], labels_train):.2f} '
        f'Val ACC {evaluator(logits[obs_idx_val], labels_val):.2f} '
        f'Test_tran ACC {evaluator(logits[obs_idx_test], labels_test_tran):.2f}')
    log_target = logits.log_softmax(dim=1)

    probs = F.softmax(logits, dim=1)
    metapaths = conf['metapaths']
    min_ns = conf['min_ns']
    reliable_src_list = []
    reliable_dst_list = []
    reliable_score_list = []
    reliable_node_indices = None
    for metapath, min_n in zip(metapaths, min_ns):
        reliable_src, reliable_dst, reliable_score, reliable_node_indices = reliable_edge_indices(obs_g, feats_obs,
                                                                                                  probs,
                                                                                                  labels_obs,
                                                                                                  obs_idx_train,
                                                                                                  obs_idx_val,
                                                                                                  obs_idx_test,
                                                                                                  metapath, p=conf['p'],
                                                                                                  min_n=min_n,
                                                                                                  return_nodes=True)
        reliable_score = torch.tensor(reliable_score, dtype=torch.float32).to(device)
        reliable_src_list.append(reliable_src)
        reliable_dst_list.append(reliable_dst)
        reliable_score_list.append(reliable_score)

    del obs_g, g, logits, probs

    # distill into MLP
    model = get_model('MLP', None, in_dim, num_classes, conf).to(device)

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    log(f'model size: {size_all_mb:.3f}MB')

    # loss_fcn1 = torch.nn.NLLLoss()
    # loss_fcn2 = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    optimizer = optim.Adam(model.parameters(),
                           lr=conf['learning_rate'], weight_decay=conf['weight_decay'])

    best_epoch = 0
    best_acc_val = 0
    training_start = time.time()

    for epoch in range(1, conf['max_epoch'] + 1):
        epoch_start = time.time()

        model.train()
        logits = model(None, feats_obs)
        log_probs = F.log_softmax(logits, dim=1)
        loss_ce = F.nll_loss(log_probs[obs_idx_train], labels_train)

        # loss_kl = F.kl_div(log_probs, log_target, reduction='batchmean', log_target=True)
        loss_kl = F.kl_div(log_probs[reliable_node_indices], log_target[reliable_node_indices], reduction='batchmean',
                           log_target=True)

        loss_nd = 0.
        for reliable_src, reliable_dst, reliable_score in zip(reliable_src_list, reliable_dst_list,
                                                              reliable_score_list):
            kl = F.kl_div(log_probs[reliable_dst], log_target[reliable_src], reduction='none', log_target=True)
            kl = (kl.sum(dim=1) * reliable_score)
            loss_nd += (kl.sum() + loss_kl * log_probs.shape[0]) / (reliable_src.shape[0] + log_probs.shape[0])
        loss_nd /= len(reliable_src_list)

        # loss = lamda * loss_ce + (1 - lamda) * loss_kl
        loss = lamda * loss_ce + (1 - lamda) * (loss_kl + gamma * loss_nd)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'CE={loss_ce:.4f} KL={loss_kl:.4f} ND={loss_nd:.4f}')

        if epoch % conf['eval_interval'] == 0:
            acc_train = eval_mlp(model, feats_train, labels_train, evaluator)
            acc_val = eval_mlp(model, feats_val, labels_val, evaluator)
            acc_test_tran = eval_mlp(model, feats_test_tran, labels_test_tran, evaluator)
            acc_test_ind = eval_mlp(model, feats_test_ind, labels_test_ind, evaluator)
            acc_test = (1 - conf['split_rate']) * acc_test_tran + conf['split_rate'] * acc_test_ind

            if best_acc_val < acc_val:
                best_epoch = epoch
                best_acc_val = acc_val
                if conf['model_save']:
                    torch.save(model.state_dict(), conf['student_path'])
                else:
                    best_state = copy.deepcopy(model.state_dict())
            time_taken = str(datetime.timedelta(seconds=(time.time() - epoch_start)))
            log(
                f'Epoch {epoch:03d} | Loss={loss:.4f} | Train={acc_train:.2f} | Val={acc_val:.2f} '
                f'Test={acc_test:.2f} Tran={acc_test_tran :.2f} Ind={acc_test_ind :.2f} | Time {time_taken}'
            )

    if conf['model_save']:
        model.load_state_dict(torch.load(conf['student_path']))
    else:
        model.load_state_dict(best_state)
    acc_test_tran = eval_mlp(model, feats_test_tran, labels_test_tran, evaluator)
    acc_test_ind = eval_mlp(model, feats_test_ind, labels_test_ind, evaluator)
    acc_test = (1 - conf['split_rate']) * acc_test_tran + conf['split_rate'] * acc_test_ind

    time_taken = str(datetime.timedelta(seconds=(time.time() - training_start)))
    log(f'Best Epoch {best_epoch:3d} | Test={acc_test :.2f} '
        f'Tran={acc_test_tran :.2f} Ind={acc_test_ind :.2f}| Total time taken {time_taken}')

    return acc_test, acc_test_tran, acc_test_ind
