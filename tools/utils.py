#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

import scipy
import os
import sys
import json
import random
import torch
import numpy as np
from pathlib import Path
import torch.nn as nn
import wandb
from loguru import logger
import torch.distributed as dist
from sentence_transformers import util
import sed_scores_eval
from sed_scores_eval.utils.scores import create_score_dataframe

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_logger(exp_name):
    log_output_dir = Path('outputs', exp_name, 'logging')
    model_output_dir = Path('outputs', exp_name, 'models')
    log_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stdout, format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)
    logger.add(log_output_dir.joinpath('output.txt'), format='{time: YYYY-MM-DD at HH:mm:ss} | {message}', level='INFO',
               filter=lambda record: record['extra']['indent'] == 1)

    return model_output_dir, log_output_dir


def setup_seed(seed):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_for_distributed(is_master):
    """
       This function disables printing when not in master process
       """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        print(os.environ.keys())
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, "env://", args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method="env://",
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)


def log_results(results, dataset, main_logger, test=False):
    if test:
        pre = "test"
    else:
        pre = "val"
    main_logger.info('{}: Caption to audio: r1: {:.2f}, r5: {:.2f}, '
                     'r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}, mAP10: {:.3f}'.format(dataset, *results["t2a"]))
    main_logger.info('{}: Audio to caption: r1: {:.2f}, r5: {:.2f}, '
                     'r10: {:.2f}, r50: {:.2f}, medr: {:.2f}, meanr: {:.2f}, mAP10: {:.3f}'.format(dataset, *results["a2t"]))
    wandb.log({
        f"{dataset}:{pre}_t2a/r1": results["t2a"][0],
        f"{dataset}:{pre}_t2a/r5": results["t2a"][1],
        f"{dataset}:{pre}_t2a/r10": results["t2a"][2],
        f"{dataset}:{pre}_t2a/mAP10": results["t2a"][-1],
    })

    wandb.log({
        f"{dataset}:{pre}_a2t/r1": results["a2t"][0],
        f"{dataset}:{pre}_a2t/r5": results["a2t"][1],
        f"{dataset}:{pre}_a2t/r10": results["a2t"][2],
        f"{dataset}:{pre}_a2t/mAP10": results["a2t"][-1],
    })


def remove_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def a2t(audio_embs, cap_embs, return_ranks=False):
    # audio to caption retrieval
    num_audios = int(audio_embs.shape[0] / 5)

    ranks = np.zeros(num_audios)
    top1 = np.zeros(num_audios)
    AP10 = np.zeros(num_audios)
    for index in range(num_audios):
        # get query audio
        audio = audio_embs[5 * index]

        # compute scores
        # d = audio @ cap_embs.T
        d = util.cos_sim(torch.Tensor(audio), torch.Tensor(cap_embs)).squeeze(0).numpy()
        inds = np.argsort(d)[::-1]

        inds_map = []

        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
            if tmp < 10:
                inds_map.append(tmp + 1)
        inds_map = np.sort(np.array(inds_map))
        # calculate average precision
        if len(inds_map) != 0:
            AP10[index] = np.sum((np.arange(1, len(inds_map) + 1) / inds_map)) / 5
        else:
            AP10[index] = 0.
        ranks[index] = rank
        top1[index] = inds[0]
    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * np.sum(AP10) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, mAP10, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr, mAP10


def t2a(audio_embs, cap_embs, return_ranks=False):
    # caption to audio retrieval
    num_audios = int(audio_embs.shape[0] / 5)

    audios = np.array([audio_embs[i] for i in range(0, audio_embs.shape[0], 5)])

    ranks = np.zeros(5 * num_audios)
    top1 = np.zeros(5 * num_audios)

    for index in range(num_audios):

        # get query captions
        queries = cap_embs[5 * index: 5 * index + 5]

        # compute scores
        # d = queries @ audios.T
        d = util.cos_sim(torch.Tensor(queries), torch.Tensor(audios)).numpy()

        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    mAP10 = 100.0 * np.sum(1 / (ranks[np.where(ranks < 10)[0]] + 1)) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, mAP10, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr, mAP10

def compute_psds_from_scores(
    scores,
    ground_truth_file,
    durations_file,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
    alpha_ct=0,
    alpha_st=0,
    max_efpr=100,
    num_jobs=4,
    save_dir=None,
):
    psds, psd_roc, single_class_rocs, *_ = sed_scores_eval.intersection_based.psds(
        scores=scores, ground_truth=ground_truth_file,
        audio_durations=durations_file,
        dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold, alpha_ct=alpha_ct, alpha_st=alpha_st,
        max_efpr=max_efpr, num_jobs=num_jobs,
    )
    return psds

def compute_collar_f1(    
    scores,
    ground_truth_file,
    collar = 0.2,
    offset_collar_rate = 0.2,
    time_decimals = 30):
    
    f_best, p_best, r_best, thresholds_best, stats_best = sed_scores_eval.collar_based.best_fscore(
        scores=scores,
        ground_truth=ground_truth_file,
        onset_collar=collar, offset_collar=collar,
        offset_collar_rate=offset_collar_rate,
        time_decimals=time_decimals,
        num_jobs=8,
    )
    return f_best

def compute_seg_f1(    
    scores,
    ground_truth_file,
    durations_file):
    
    f_best, p_best, r_best, thresholds_best, stats_best = sed_scores_eval.segment_based.best_fscore(
        scores=scores,
        ground_truth=ground_truth_file,
        audio_durations=durations_file,
        num_jobs=8,
    )
    return f_best
    
def post_process_sed(c_scores, classes):
    _, nf, nc = c_scores.size()
    c_scores = c_scores.squeeze(0).detach().cpu().numpy()
    scores_raw = create_score_dataframe(
            scores=c_scores,
            timestamps=[i * 10 / nf for i in range(nf + 1)],
            event_classes=classes,
    )
        
    c_scores = scipy.ndimage.filters.median_filter(c_scores, (3, 1))
    scores_postprocessed = create_score_dataframe(
        scores=c_scores,
        timestamps=[i * 10 / nf for i in range(nf + 1)],
        event_classes=classes,
    )
    return scores_raw, scores_postprocessed