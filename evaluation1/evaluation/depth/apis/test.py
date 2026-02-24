# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings

import mmcv
import numpy as np
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from imageio import imread
def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix=".npy", delete=False, dir=tmpdir
        ).name
    np.save(temp_file_name, array)
    return temp_file_name


def replace_str(s):
    if s[0] == "/":
        return s[1:]
    new_str = s.replace("/", "_")
    return new_str


def single_gpu_test(
    model,
    data_loader,    
    pre_eval=False,
    format_only=False,
    format_args={},
):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    # when none of them is set true, return depth results as
    # a list of np.array.
    assert [pre_eval, format_only].count(True) <= 1, (
        "``pre_eval`` and ``format_only`` are mutually "
        "exclusive, only one of them could be true ."
    )

    model.eval()
    results = []
    logs = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    log_num_images = 20
    log_interval = int(len(data_loader) / log_num_images)
    for i, (batch_indices, data) in enumerate(zip(loader_indices, data_loader)):    
        result = [None]
        with torch.no_grad():
            result_depth = model(return_loss=False, **data)
        if i%log_interval==0:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            img = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])[0]            
            gt = imread(dataset.img_infos[batch_indices[0]]['ann']['depth_map'])
            pred = result_depth[0][0]  
            #### for vggt ###          
            # pred = torch.nn.functional.interpolate(torch.from_numpy(result_depth[0]).permute(0,3,1,2), size=gt.shape, mode='bilinear', align_corners=False).permute(0,2,3,1)[0]
            #######################  
            # remove mark after finish          
            logs.append({'img': img, 'ground_truth': gt, 'pred': pred, 'img_metas': img_metas[0]})            
            # import os
            # output_dir = './da2_dave'
            # os.makedirs(output_dir, exist_ok=True)
            # torch.save({'img': img, 'ground_truth': gt, 'pred': pred, 'img_metas': img_metas[0]}, osp.join(output_dir, replace_str(img_metas[0]['ori_filename'])+'.pth'))


        if format_only:
            result = dataset.format_results(
                result_depth, indices=batch_indices, **format_args
            )
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            # eval metric, result depth.
            result, result_depth = dataset.pre_eval(result_depth, indices=batch_indices)

        # if format only, result will be formated output
        # if pre_eval, result will be pre_eval res for final aggregation
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results, logs


def multi_gpu_test(
    model,
    data_loader,
    tmpdir=None,
    gpu_collect=False,
    pre_eval=False,
    format_only=False,
    format_args={},
):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval. Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """

    # when none of them is set true, return depth estimation results as
    # a list of np.array.
    assert [pre_eval, format_only].count(True) <= 1, (
        "``pre_eval`` and ``format_only`` are mutually "
        "exclusive, only one of them could be true ."
    )

    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        result = None

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args
            )

        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result, _ = dataset.pre_eval(result, indices=batch_indices)

        results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results
