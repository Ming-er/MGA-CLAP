#!/usr/bin/env python3
# coding: utf-8
# @Author  : Yiming Li @ ICT, CAS
# @E-mail  : liyiming22s1@ict.ac.cn

from torch.cuda.amp import GradScaler
import time
from pprint import PrettyPrinter
import torch
import argparse
import ruamel.yaml as yaml
from tqdm import tqdm
from loguru import logger
from data_handling.datamodule import AudioCaptionDataModule
from data_handling.pretrain_dataset import pretrain_dataloader
from models.ase_model import ASE
import torch.distributed as dist
from tools.optim_utils import get_optimizer, cosine_lr
from tools.utils import *
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


def train(model, dataloader, optimizer, scheduler, device, epoch, scaler):
    model.train()
    autocast = torch.cuda.amp.autocast
    epoch_loss = AverageMeter()
    start_time = time.time()

    if is_dist_avail_and_initialized():
        dataloader.sampler.set_epoch(epoch)

    for batch_id, (audio, text, idx) in tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()
        step = len(dataloader) * (epoch - 1) + batch_id
        scheduler(step)

        audio = audio.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        with autocast():
            loss = model(audio, text, idx)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss.update(loss.cpu().item())

    elapsed_time = time.time() - start_time

    return {
        "loss": epoch_loss.avg,
        "time": elapsed_time
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="settings/pretrain.yaml", type=str,
                        help="Setting files")
    parser.add_argument("-n", "--exp_name", default="exp_name", type=str,
                        help="name of this experiment.")
    parser.add_argument("-l", "--lr", default=5e-5, type=float,
                        help="Learning rate.")
    parser.add_argument("-t", "--model_type", default="cnn", type=str,
                        help="Model type.")
    parser.add_argument("-m", "--model", default="Cnn14", type=str,
                        help="Model name.")
    parser.add_argument("-a", "--max_length", default=30, type=int,
                        help="Max length.")
    parser.add_argument("-s", "--batch_size", default=128, type=int,
                        help="Batch size.")
    parser.add_argument("-b", "--blacklist", default='blacklist_exclude_ub8k_esc50_vggsound.json', type=str,
                        help="Blacklist file.")
    parser.add_argument('--local_rank', default=-1, type=int)
    args = parser.parse_args()

    exp_name = args.exp_name

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # setup distribution mode
    init_distributed_mode(args)
    device = config["device"]

    # setup seed
    seed = config["seed"] + get_rank()
    setup_seed(seed)

    exp_name = exp_name + f"_lr_{args.lr}_seed_{seed}"
    # create pretrain dataloader
    dataloader = pretrain_dataloader(config,
                                     bucket=True,
                                     bucket_boundaries=(5, 30, 6),
                                     is_distributed=is_dist_avail_and_initialized(),
                                     num_tasks=get_world_size(),
                                     global_rank=get_rank())
    
    
    clotho_datamodule = AudioCaptionDataModule("Clotho")
    clotho_test_loader = clotho_datamodule.test_dataloader()
    ac_datamodule = AudioCaptionDataModule("AudioCaps")
    ac_test_loader = ac_datamodule.test_dataloader()
    
    # setup model
    model = ASE(config)
    model.to(device)

    # setup optim utils
    optimizer = get_optimizer(model.parameters(),
                              lr=config["optim_args"]["lr"],
                              betas=config["optim_args"]["betas"],
                              eps=config["optim_args"]["eps"],
                              momentum=config["optim_args"]["momentum"],
                              optimizer_name=config["optim_args"]["optimizer_name"])
    scheduler = cosine_lr(optimizer,
                          base_lr=config["optim_args"]["lr"],
                          warmup_length=config["optim_args"]["warmup_epochs"] * len(dataloader),
                          steps=len(dataloader) * config["training"]["epochs"])
    start_epoch = 1
    max_epoch = config["training"]["epochs"]

    if config["resume"]:
        cp = torch.load(config.checkpoint, map_location="cpu")
        state_dict = cp["model"]

        optimizer.load_state_dict(cp["optimizer"])
        start_epoch = cp["epoch"] + 1
        model.load_state_dict(state_dict)

    # setup logger
    model_output_dir, log_output_dir = set_logger(exp_name)

    main_logger = logger.bind(indent=1)

    # print training settings
    printer = PrettyPrinter()
    main_logger.info('Training setting:\n'
                     f'{printer.pformat(config)}')

    main_logger.info(f'Total numer of parameters: {sum([i.numel() for i in model.parameters()])}')
    main_logger.info(f'Size of training set: {len(dataloader.dataset)}, size of batches: {len(dataloader)}')
    
    model_without_ddp = model
    if is_dist_avail_and_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True
        )
        model_without_ddp = model.module

    desed_stats = []
    clotho_stats = []
    ac_stats = []

    scaler = GradScaler()
    for epoch in range(start_epoch, max_epoch + 1):
        main_logger.info(f'Training for epoch [{epoch}]')
        
        train_statics = train(model, dataloader, optimizer, scheduler, device, epoch, scaler)
        loss = train_statics["loss"]
        elapsed_time = train_statics["time"]

        main_logger.info(f'Training statistics:\tloss for epoch [{epoch}]: {loss:.3f},'
                         f'\ttime: {elapsed_time:.1f}, lr: {optimizer.param_groups[0]["lr"]:.6f}.')

        if is_dist_avail_and_initialized():
            dist.barrier()
            torch.cuda.empty_cache()
        
        # validate on Clotho
        if is_main_process():
            clotho_metrics = validate_re(model_without_ddp, clotho_test_loader, device)
            main_logger.info(f'Clotho statistics for epoch [{epoch}]:\t t2a-r1: {clotho_metrics["t2a"][0]:.3f}, t2a-r5: {clotho_metrics["t2a"][1]:.3f}, a2t-r1: {clotho_metrics["a2t"][0]:.3f}, a2t-r5: {clotho_metrics["a2t"][1]:.3f}.')
            clotho_stats.append(clotho_metrics["t2a"][0] + clotho_metrics["t2a"][1] + clotho_metrics["a2t"][0] + clotho_metrics["a2t"][1])
            if clotho_stats[-1] >= max(clotho_stats):
                sav_obj = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch
                }
                torch.save(sav_obj, str(model_output_dir) + "/clotho_best_model.pt")
                
        # validate on AudioCaps
        if is_main_process():
            ac_metrics = validate_re(model_without_ddp, ac_test_loader, device)
            main_logger.info(f'AudioCaps statistics for epoch [{epoch}]:\t t2a-r1: {ac_metrics["t2a"][0]:.3f}, t2a-r5: {ac_metrics["t2a"][1]:.3f}, a2t-r1: {ac_metrics["a2t"][0]:.3f}, a2t-r5: {ac_metrics["a2t"][1]:.3f}.')
            ac_stats.append(ac_metrics["t2a"][0] + ac_metrics["t2a"][1] + ac_metrics["a2t"][0] + ac_metrics["a2t"][1])
            if ac_stats[-1] >= max(ac_stats):
                sav_obj = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch
                }
                torch.save(sav_obj, str(model_output_dir) + "/ac_best_model.pt")
                
        if is_main_process() and epoch > 2 and epoch < 10:
            sav_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "epoch": epoch
            }
            torch.save(sav_obj, str(model_output_dir) + "/model_" + str(epoch) + ".pt")
            
        if is_dist_avail_and_initialized():
            dist.barrier()
            torch.cuda.empty_cache()

    main_logger.info("Done.")


@torch.no_grad()
def validate_sed(model, dataloader, sed_classes, raw_sed_classes, sed_gt, sed_dur, device, name):
    model.eval()
    _, word_embeds, attn_mask = model.encode_text(sed_classes)
    _, text_embeds, _ = model.msc(word_embeds, model.codebook, attn_mask)
    text_embeds = F.normalize(text_embeds, dim=-1)
    scores_raw_dic, scores_postprocessed_dic = {}, {}
    for _, (audio, filename) in tqdm(enumerate(dataloader), total=len(dataloader)):
        audio = audio.to(device, non_blocking=True)
        _, frame_embeds = model.encode_audio(audio)
        _, frame_embeds, _ = model.msc(frame_embeds.unsqueeze(1), model.codebook)
        frame_embeds = F.normalize(frame_embeds, dim=-1)
        similarity = frame_embeds @ text_embeds.t()
        scores_raw, scores_postprocessed = post_process_sed(similarity, raw_sed_classes)
        scores_raw_dic[filename[0].split(".wav")[0]] = scores_raw
        scores_postprocessed_dic[filename[0].split(".wav")[0]] = scores_postprocessed
        
    pop_lst = []

    for k in scores_postprocessed_dic.keys():
        if k not in sed_gt.keys():
            pop_lst.append(k)
    for k in pop_lst:       
            scores_postprocessed_dic.pop(k)
     
    psds1_sed_scores_eval = compute_psds_from_scores(
        scores_postprocessed_dic,
        sed_gt,
        sed_dur,
        dtc_threshold=0.7,
        gtc_threshold=0.7,
        cttc_threshold=None,
        alpha_ct=0,
        alpha_st=1,
    )

    psds2_sed_scores_eval = compute_psds_from_scores(
        scores_postprocessed_dic,
        sed_gt,
        sed_dur,
        dtc_threshold=0.1,
        gtc_threshold=0.1,
        cttc_threshold=0.3,
        alpha_ct=0.5,
        alpha_st=1,
    )

    eb_f1 = compute_collar_f1(scores_postprocessed_dic, sed_gt)
    seg_f1 = compute_seg_f1(scores_postprocessed_dic, sed_gt, sed_dur)

    return psds1_sed_scores_eval, psds2_sed_scores_eval, eb_f1["macro_average"], seg_f1["macro_average"]

@torch.no_grad()
def validate_re(model, dataloader, device):
    model.eval()
    audio_embeds_all, text_embeds_all = [], []
    for batch_idx, (audio, text, idx) in tqdm(enumerate(dataloader), total=len(dataloader)):
        audio = audio.to(device)

        _, frame_embeds = model.encode_audio(audio)
        audio_embeds = model.msc(frame_embeds, model.codebook)
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        
        _, word_embeds, attn_mask = model.encode_text(text)
        text_embeds = model.msc(word_embeds, model.codebook, attn_mask)
        text_embeds = F.normalize(text_embeds, dim=-1)

        audio_embeds_all.append(audio_embeds.cpu())
        text_embeds_all.append(text_embeds.cpu())

    audio_embeds_all = torch.cat(audio_embeds_all, dim=0).numpy()
    text_embeds_all = torch.cat(text_embeds_all, dim=0).numpy()

    # evaluate text to audio retrieval
    r1, r5, r10, r50, medr, meanr, mAP10 = t2a(audio_embeds_all, text_embeds_all)

    # evaluate audio to text retrieval
    r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a = a2t(audio_embeds_all, text_embeds_all)

    return {"t2a": [r1, r5, r10, r50, medr, meanr, mAP10],
            "a2t": [r1_a, r5_a, r10_a, r50_a, medr_a, meanr_a, mAP10_a]}


if __name__ == '__main__':
    main()
