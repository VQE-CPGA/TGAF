import os
import time
import yaml
import argparse
# import torch
import os.path as op
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import utils_MS  # my tool box
import mfqev2_MS
from net_SPL import TGDA_7


def receive_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='/share3/home/zqiang/STDF30/config/TGDA/option_TGDA_7_22.yml', 
        help='Path to option YAML file.')
    args = parser.parse_args()
    
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils_MS.get_timestr()

    opts_dict['train']['checkpoint_save_path_pre'] = op.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
        )
    opts_dict['test']['restore_iter'] = int(
        opts_dict['test']['restore_iter']
        )
    opts_dict['test']['checkpoint_save_path'] = (
        f"{opts_dict['train']['checkpoint_save_path_pre']}"
        f"{opts_dict['test']['restore_iter']}"
        '.pt'
        )

    return opts_dict


def main():
    # ==========
    # parameters
    # ==========

    opts_dict = receive_arg()
    unit = opts_dict['test']['criterion']['unit']
    PSNRS = []
    SSIMS = []
    
    # create datasets
    test_ds_type = opts_dict['dataset']['test']['type']
    radius = opts_dict['network']['radius']
    assert test_ds_type in mfqev2_MS.__all__, \
        "Not implemented!"
    test_ds_cls = getattr(mfqev2_MS, test_ds_type)
    test_ds = mfqev2_MS.VideoTestMFQEv2Dataset(opts_dict=opts_dict['dataset']['test'], radius=radius)

    test_num = len(test_ds)
    test_vid_num = test_ds.get_vid_num()

    # create datasamplers
    test_sampler = None  # no need to sample test data

    # create dataloaders
    test_loader = mindspore.dataset.GeneratorDataset(source=test_ds,column_names=["data", "label"],
    num_parallel_workers=opts_dict['dataset']['val']['num_worker_per_gpu'], sampler=test_sampler,
    python_multiprocessing=True, max_rowsize=6) 

    # ==========
    # create & load model
    # ==========

    model = TGDA_7()
    checkpoint_save_path = opts_dict['test']['checkpoint_save_path']
    checkpoint = torch.load(checkpoint_save_path,map_location='cpu')     ###### ??????
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])
    
    model = model.cuda()
    model.eval()


    # define criterion
    criterion = utils_MS.PSNR()

    # ==========
    # validation
    # ==========

    # create counters
    per_aver_dict = dict()
    ori_aver_dict = dict()
    name_vid_dict = dict()
    per_aver_dict_ssim = dict()
    ori_aver_dict_ssim = dict()
    for index_vid in range(test_vid_num):
        per_aver_dict[index_vid] = utils_MS.Counter()
        ori_aver_dict[index_vid] = utils_MS.Counter()
        per_aver_dict_ssim[index_vid] = utils_MS.Counter()
        ori_aver_dict_ssim[index_vid] = utils_MS.Counter()
        name_vid_dict[index_vid] = ""

    pbar = tqdm(
        total=test_num, 
        ncols=opts_dict['test']['pbar_len']
        )

    # fetch the first batch
    test_prefetcher.reset()
    val_data = test_prefetcher.next()

    with torch.no_grad():       #######  ????? 
        while val_data is not None:
            # get data
            gt_data = val_data['gt'].cuda()  # (B [RGB] H W)
            lq_data = val_data['lq'].cuda()  # (B T [RGB] H W)
            index_vid = val_data['index_vid'].item()
            name_vid = val_data['name_vid'][0]  # bs must be 1!
            
            b, _, c, _, _  = lq_data.shape
            assert b == 1, "Not supported!"
            
            input_data = mindspore.ops.cat(
                [lq_data[:,:,i,...] for i in range(c)], 
                dim=1
                )  # B [R1 ... R7 G1 ... G7 B1 ... B7] H W
            enhanced_data = model(input_data)  # (B [RGB] H W)

            # eval
            batch_ori = criterion(lq_data[0, radius, ...], gt_data[0])
            batch_perf = criterion(enhanced_data[0], gt_data[0])
            enhanced_data = enhanced_data[0].cpu().squeeze().numpy()
            lq_data2 = lq_data[0, radius, ...].cpu().squeeze().numpy()
            gt_data2 = gt_data[0].cpu().squeeze().numpy()
            
            batch_ori_ssim = structural_similarity(lq_data2, gt_data2, data_range=1.0)
            batch_perf_ssim = structural_similarity(enhanced_data, gt_data2, data_range=1.0)
                    
            # display
            pbar.set_description(
                "{:s}: [{:.3f}] {:s} -> [{:.3f}] {:s}"
                .format(name_vid, batch_ori, unit, batch_perf, unit)
                )
            pbar.update()

            # log
            per_aver_dict[index_vid].accum(volume=batch_perf)
            ori_aver_dict[index_vid].accum(volume=batch_ori)
            per_aver_dict_ssim[index_vid].accum(volume=batch_perf_ssim)
            ori_aver_dict_ssim[index_vid].accum(volume=batch_ori_ssim)
            if name_vid_dict[index_vid] == "":
                name_vid_dict[index_vid] = name_vid
            else:
                assert name_vid_dict[index_vid] == name_vid, "Something wrong."

            # fetch next batch
            val_data = test_prefetcher.next()
        
    # end of val
    pbar.close()

    # log
    msg = '\n' + '<' * 10 + ' Results ' + '>' * 10
    print(msg)
    for index_vid in range(test_vid_num):
        per = per_aver_dict[index_vid].get_ave()
        ori = ori_aver_dict[index_vid].get_ave()
        name_vid = name_vid_dict[index_vid]
        msg = "{:s}: [{:.3f}] {:s} -> [{:.3f}] {:s} Delta:[{:.3f}] ".format(
            name_vid, ori, unit, per, unit, per-ori
            )
        print(msg)
        per_ssim = per_aver_dict_ssim[index_vid].get_ave()
        ori_ssim = ori_aver_dict_ssim[index_vid].get_ave()
        msg = "SSIM: {:s}: [{:.3f}] {:s} -> [{:.3f}] {:s} Delta:[{:.3f}] ".format(
            name_vid, ori_ssim*100.0, unit, per_ssim*100.0, unit,(per_ssim-ori_ssim)*100.0
            )
        print(msg)
    ave_per = np.mean([
        per_aver_dict[index_vid].get_ave() for index_vid in range(test_vid_num)
        ])
    ave_ori = np.mean([
        ori_aver_dict[index_vid].get_ave() for index_vid in range(test_vid_num)
        ])
    ave_per_ssim = np.mean([
            per_aver_dict_ssim[index_vid].get_ave() for index_vid in range(test_vid_num)
            ])
    ave_ori_ssim = np.mean([
        ori_aver_dict_ssim[index_vid].get_ave() for index_vid in range(test_vid_num)
        ])
    msg = (
        f"{'> ori: [{:.3f}] {:s}'.format(ave_ori, unit)}\n"
        f"{'> ave: [{:.3f}] {:s}'.format(ave_per, unit)}\n"
        f"{'> delta: [{:.3f}] {:s}'.format(ave_per - ave_ori, unit)}"
        )
    print(msg)
    msg = (
            f"{'> ori: [{:.3f}] {:s}'.format(ave_ori_ssim*100.0, unit)}\n"
            f"{'> ave: [{:.3f}] {:s}'.format(ave_per_ssim*100.0, unit)}\n"
            f"{'> delta: [{:.3f}] {:s}'.format(ave_per_ssim*100.0 - ave_ori_ssim*100.0, unit)}"
            )
    print(msg)

    # ==========
    # final log & close logger
    # ==========

    msg1 = "Number of Parameters: [{:.1f}]".format(sum([np.prod(p.size()) for p in model.parameters()]))
    print(msg)
    print(msg1)
    
    msg = (
        f"\n{'<' * 10} Goodbye {'>' * 10}\n"
        f"Timestamp: [{utils_MS.get_timestr()}]"
        )
    print(msg)
    

if __name__ == '__main__':
    main()
    