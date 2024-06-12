import os
import math
import time
import yaml
import argparse
import os.path as op
import numpy as np
from tqdm import tqdm
# from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
import utils_MS  # my tool box
import mfqev2_MS
from net_SPL import TGDA_7
import mindspore
from mindspore import context, TimeMonitor, Model, load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor



def receive_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local_rank', type=int, default=0, 
        help='Distributed launcher requires.'
        )
    args = parser.parse_args()
    
    with open('/share3/home/zqiang/STDF30/config/TGDA/option_TGDA_7_22.yml', 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['train']['rank'] = args.local_rank

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils_MS.get_timestr()

    opts_dict['train']['log_path'] = op.join(
        "exp", opts_dict['train']['exp_name'], "log.log"
        )
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
        )
    opts_dict['train']['num_gpu'] = 1
    # opts_dict['train']['num_gpu'] = torch.cuda.device_count()
    # if opts_dict['train']['num_gpu'] > 1:
    #     opts_dict['train']['is_dist'] = True
    # else:
    #     opts_dict['train']['is_dist'] = False
    
    opts_dict['test']['restore_iter'] = int(
        opts_dict['test']['restore_iter']
        )

    return opts_dict


def main():
    # ==========
    # parameters
    # ==========

    opts_dict = receive_arg()
    rank = opts_dict['train']['rank']
    unit = opts_dict['train']['criterion']['unit']
    num_iter = int(opts_dict['train']['num_iter'])
    interval_print = int(opts_dict['train']['interval_print'])
    interval_val = int(opts_dict['train']['interval_val'])
    
    # ==========
    # init distributed training
    # ==========

    if opts_dict['train']['is_dist']:
        utils_MS.init_dist(local_rank=rank, backend='nccl')

    seed = opts_dict['train']['random_seed']
    # >I don't know why should rs + rank
    utils_MS.set_random_seed(seed + rank)

   
    # create datasets
    train_ds_type = opts_dict['dataset']['train']['type']
    val_ds_type = opts_dict['dataset']['val']['type']
    radius = opts_dict['network']['radius']
    # train_ds_cls = getattr(mfqev2_MS, train_ds_type)
    # val_ds_cls = getattr(mfqev2_MS, val_ds_type)
    train_ds = mfqev2_MS.MFQEv2Dataset(opts_dict=opts_dict['dataset']['train'], radius=radius)
    val_ds = mfqev2_MS.VideoTestMFQEv2Dataset(opts_dict=opts_dict['dataset']['val'], radius=radius)

    # print('',train_ds.get_dataset_size())
    
    # create datasamplers
    train_sampler = mindspore.dataset.DistributedSampler(num_shards=opts_dict['train']['num_gpu'],shard_id=rank)
    val_sampler = None  # no need to sample val data
    batch_size = opts_dict['dataset']['train']['batch_size_per_gpu']
    train_ds = train_ds.batch(batch_size, drop_remainder=False) # dataset = dataset.batch(5, True)

    # create dataloaders
    train_loader = mindspore.dataset.GeneratorDataset(source=train_ds,column_names=["data", "label"],
    num_parallel_workers=opts_dict['dataset']['train']['num_worker_per_gpu'], sampler=train_sampler,
    python_multiprocessing=True, max_rowsize=6)  
    
    
    
    val_loader = mindspore.dataset.GeneratorDataset(source=val_ds,column_names=["data", "label"],
    num_parallel_workers=opts_dict['dataset']['train']['num_worker_per_gpu'], sampler=val_sampler,
    python_multiprocessing=True, max_rowsize=6) 
 
    # assert train_loader is not None
   
    num_iter_per_epoch = math.ceil(len(train_ds) * \
        opts_dict['dataset']['train']['enlarge_ratio'] / batch_size)
    num_epoch = math.ceil(num_iter / num_iter_per_epoch)
    val_num = len(val_ds)
    
    # create dataloader prefetchers
    tra_prefetcher = train_loader # utils_MS.CPUPrefetcher(train_loader)
    val_prefetcher = val_loader # utils_MS.CPUPrefetcher(val_loader)

    model = TGDA_7()
    # print("Number of Parameters: ", sum([np.prod(p.size()) for p in model.parameters()]))
    optimizer = mindspore.nn.Adam(params=model.trainable_params())

    
    
    model = Model(model, loss_fn=utils_MS.CharbonnierLoss(**opts_dict['train']['loss']), optimizer=optimizer, metrics=None)
    # ==========
    # define loss func & optimizer & scheduler & scheduler & criterion
    # ==========

    start_iter = 0  # should be restored
    start_epoch = start_iter // num_iter_per_epoch

    # display and log
    if rank == 0:
        msg = (
            f"\n{'<' * 10} Dataloader {'>' * 10}\n"
            f"total iters: [{num_iter}]\n"
            f"total epochs: [{num_epoch}]\n"
            f"iter per epoch: [{num_iter_per_epoch}]\n"
            f"val sequence: [{val_num}]\n"
            f"start from iter: [{start_iter}]\n"
            f"start from epoch: [{start_epoch}]"
            )
        print(msg)

    # ==========
    # start training + validation (test)
    # ==========

    config_ck = CheckpointConfig(save_checkpoint_steps=500,keep_checkpoint_max=100)  #  args.keep_checkpoint_max
    time_cb = TimeMonitor()
    ckpoint_cb = ModelCheckpoint(prefix="ckpt_" + str(rank), directory=(f"{opts_dict['train']['checkpoint_save_path_pre']}"f"{rank}"),
                                 config=config_ck)
    loss_cb = LossMonitor()
    # eval_cb = EvaluateCallBack(model, eval_dataset=data.val_dataset, src_url=ckpt_save_dir,
    #                            train_url=os.path.join(args.train_url, "ckpt_" + str(rank)),
    #                            save_freq=args.save_every)
    model.train(num_epoch, train_ds, callbacks=[time_cb, ckpoint_cb, loss_cb])

    num_iter_accum = start_iter
    for current_epoch in range(start_epoch, num_epoch + 1):
        # shuffle distributed subsamplers before each epoch
        if opts_dict['train']['is_dist']:
            train_sampler.set_epoch(current_epoch)

        # train this epoch
        while train_data is not None:
            # over sign
            num_iter_accum += 1
            if num_iter_accum > num_iter:
                break

            # get data
            gt_data = train_data['gt'].to(rank)  # (B [RGB] H W)
            lq_data = train_data['lq'].to(rank)  # (B T [RGB] H W)
            b, _, c, _, _  = lq_data.shape
            input_data = mindspore.ops.cat([lq_data[:,:,i,...] for i in range(c)], dim=1)  # B [R1 ... R7 G1 ... G7 B1 ... B7] H W
            enhanced_data = model(input_data)
            # get loss
            optimizer.zero_grad()  # zero grad
            loss = mindspore.ops.mean(mindspore.ops.stack([loss_func(enhanced_data[i], gt_data[i]) for i in range(b)]))  # cal loss
            loss.backward()  # cal grad
            optimizer.step()  # update parameters

            # update learning rate
            if opts_dict['train']['scheduler']['is_on']:
                scheduler.step()  # should after optimizer.step()

            if (num_iter_accum % interval_print == 0) and (rank == 0):
                # display & log
                lr = optimizer.param_groups[0]['lr']
                loss_item = loss.item()

            if ((num_iter_accum % interval_val == 0) or \
                (num_iter_accum == num_iter)) and (rank == 0):
                # save model
                checkpoint_save_path = (f"{opts_dict['train']['checkpoint_save_path_pre']}"f"{num_iter_accum}"".pt")
                state = {'num_iter_accum': num_iter_accum,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict(),  }
                if opts_dict['train']['scheduler']['is_on']:
                    state['scheduler'] = scheduler.state_dict()
                torch.save(state, checkpoint_save_path)    ######  ????? 
                
                # validation
                with torch.no_grad():     ######  ????? 
                    per_aver_dict = {}
                    for index_vid in range(vid_num):
                        per_aver_dict[index_vid] = utils_MS.Counter()
                    pbar = tqdm(total=val_num, ncols=opts_dict['train']['pbar_len'])
                
                    # train -> eval
                    model.eval()

                    # fetch the first batch
                    val_prefetcher.reset()
                    val_data = val_prefetcher.next()
                    
                    while val_data is not None:
                        # get data
                        gt_data = val_data['gt'].to(rank)  # (B [RGB] H W)
                        lq_data = val_data['lq'].to(rank)  # (B T [RGB] H W)
                        index_vid = val_data['index_vid'].item()
                        name_vid = val_data['name_vid'][0]  # bs must be 1!
                        b, _, c, _, _  = lq_data.shape
                        input_data = mindspore.ops.cat(
                            [lq_data[:,:,i,...] for i in range(c)], 
                            dim=1
                            )  # B [R1 ... R7 G1 ... G7 B1 ... B7] H W
                        enhanced_data = model(input_data)  # (B [RGB] H W)

                        # eval
                        batch_perf = np.mean([criterion(enhanced_data[i], gt_data[i]) for i in range(b)]) # bs must be 1!

                        # display
                        pbar.set_description("{:s}: [{:.3f}] {:s}".format(name_vid, batch_perf, unit))
                        pbar.update()

                        # log
                        per_aver_dict[index_vid].accum(volume=batch_perf)

                        # fetch next batch
                        val_data = val_prefetcher.next()
                    
                    # end of val
                    pbar.close()

                    # eval -> train
                    model.train()

                # log
                ave_per = np.mean([
                    per_aver_dict[index_vid].get_ave() for index_vid in range(vid_num)
                    ])
                msg = (
                    "> model saved at {:s}\n"
                    "> ave val per: [{:.3f}] {:s}"
                    ).format(
                        checkpoint_save_path, ave_per, unit
                        )
                print(msg)


if __name__ == '__main__':
    main()
    
