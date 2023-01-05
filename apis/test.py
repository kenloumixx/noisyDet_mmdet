# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
from typing import Optional, List

import torch
import torch.nn.functional as F
import time 


def total_tensor(tensor, max_num):      # 1. 패딩 2. 이어붙이기 3. 패딩 벗기기 -> max num 은 딱 패딩하기 좋은 사이즈로만..!
    rank, world_size = get_dist_info()
    # 1. padding하기 
    delta = max_num - tensor.size(0)            # batch 이후로 가장 앞단의 tensor size 가져오기
    # delta만큼 concat
    tensor = torch.cat([tensor, torch.zeros_like(tensor)[:delta]])   # 전부 같은 사이즈
    
    # 2. 이어붙이기
    output = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(output, tensor) # -> output = [tensor, tensor, ..., tensor]
    
    # 3. delta값들 가져오기
    total_delta = [None for _ in range(world_size)]
    rank_delta = [delta]    # 합치고 싶은 정보를 담아놓기
    all_gather_object(total_delta, rank_delta)       # total_delta [309, 158, 276, 185, 0, 252, 323, 635]

    # 4. 패딩 제거하기 -> 원래 갯수만큼 나누고, 각각에서 pad를 제거한 다음 다시 concat하기
    total_output = torch.cat(output)    # output stack하면 다 합쳐진거!   # total_output torch.Size([8, 4809, 81]) -> 38472
    chunked_output_list = list(torch.tensor_split(total_output, world_size))


    for idx, rank_delta in enumerate(total_delta):
        if rank_delta == 0:
            continue
        chunked_output_list[idx] = chunked_output_list[idx][:-rank_delta]

    output = torch.cat(chunked_output_list)
    return output


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def all_gather_into_tensor(output_tensor, input_tensor, group_size, group=None, async_op=False):
    output_tensor = (
        output_tensor
        if not output_tensor.is_complex()
        else torch.view_as_real(output_tensor)
    )
    input_tensor = (
        input_tensor
        if not input_tensor.is_complex()
        else torch.view_as_real(input_tensor)
    )
    group = torch.distributed.new_group(list(range(group_size)))
    work = group._allgather_base(output_tensor, input_tensor)

    if async_op:
        return work
    else:
        work.wait()




def all_gather_object(object_list, obj, group=None):
    import contextlib
    import io
    import logging
    import os
    import pickle
    import time
    import warnings
    from datetime import timedelta
    from typing import Callable, Dict, Optional, Tuple, Union
    
    _pickler = pickle.Pickler
    _unpickler = pickle.Unpickler
    
    def _tensor_to_object(tensor, tensor_size):
        buf = tensor.numpy().tobytes()[:tensor_size]
        return _unpickler(io.BytesIO(buf)).load()

    def _object_to_tensor(obj):
        f = io.BytesIO()
        _pickler(f).dump(obj)
        byte_storage = torch.ByteStorage.from_buffer(f.getvalue())  # type: ignore[attr-defined]
        # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
        # Otherwise, it will casue 100X slowdown.
        # See: https://github.com/pytorch/pytorch/issues/65696
        byte_tensor = torch.ByteTensor(byte_storage)
        local_size = torch.LongTensor([byte_tensor.numel()])
        return byte_tensor, local_size


    input_tensor, local_size = _object_to_tensor(obj)
    current_device = torch.device("cuda", torch.cuda.current_device())
    input_tensor = input_tensor.to(current_device)
    local_size = local_size.to(current_device)

    group_size = torch.distributed.get_world_size()
    object_sizes_tensor = torch.zeros(
        group_size, dtype=torch.long, device=current_device
    )
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    group = torch.distributed.new_group(list(range(group_size)))
    torch.distributed.all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(
        max_object_size * group_size, dtype=torch.uint8, device=current_device
    )

    output_tensors = [
        coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
        for i in range(group_size)
    ]
    torch.distributed.all_gather(output_tensors, input_tensor, group=group)

    k = 0
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        if tensor.device != torch.device("cpu"):
            tensor = tensor.cpu()

        tensor_size = object_size_list[i]
        output = _tensor_to_object(tensor, tensor_size)
        for j in range(len(output)):
            object_list[k] = output[j]
            k += 1    

# added
def all_gather(tensor: torch.Tensor, fixed_shape: Optional[List] = None) -> List[torch.Tensor]:
    def compute_padding(shape, new_shape):
        padding = []
        for dim, new_dim in zip(shape, new_shape):
            padding.insert(0, new_dim - dim)
            padding.insert(0, 0)
        return padding

    input_shape = tensor.shape
    if fixed_shape is not None:
        padding = compute_padding(tensor.shape, fixed_shape)
        if sum(padding) > 0:
            tensor = F.pad(tensor, pad=padding, mode='constant', value=0)
    
    output = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(output, tensor)

    all_input_shapes = None
    if fixed_shape is not None:
        # gather all shapes
        tensor_shape = torch.tensor(input_shape, device=tensor.device)
        all_input_shapes = [torch.zeros_like(tensor_shape) for _ in range(dist.get_world_size())]
        dist.all_gather(all_input_shapes, tensor_shape)
        all_input_shapes = [t.tolist() for t in all_input_shapes]

    if all_input_shapes:
        for i, shape in enumerate(all_input_shapes):
            padding = compute_padding(output[i].shape, shape)
            if sum(padding) < 0:
                output[i] = F.pad(output[i], pad=padding)

    return output






def gmm_multi_gpu_test(model, data_loader, tmpdir=None, gmm=None, gpu_collect=False, return_loss=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    dataset = data_loader.dataset
    
    # num_data = 36334  # 원래는 85xxx어쩌고였었음
    num_bbox = len(dataset.coco.anns)
    num_cls = len(dataset.coco.dataset['categories'])

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.


    bbox_ids_list = []
    loss_cls_list = []
    loss_bbox_list = []
    logits_list = []
    gt_labels_list = []

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            box_ids, cls_loss, cls_labels, bbox_loss, logits = model(return_loss=return_loss, gmm=gmm, **data)     # TODO rescale loss 옵션을 받아서.. 

        loss_cls_list.extend(cls_loss)
        loss_bbox_list.extend(bbox_loss)
        logits_list.extend(logits)
        gt_labels_list.extend(cls_labels)
        bbox_ids_list.extend(box_ids)

        if rank == 0:
            if gmm: batch_size = 1
            else: batch_size = data['img'].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    loss_cls_tensor = torch.stack(loss_cls_list)
    loss_bbox_tensor = torch.stack(loss_bbox_list)
    logits_tensor = torch.stack(logits_list)
    gt_labels_tensor = torch.stack(gt_labels_list)
    bbox_ids_tensor = torch.stack(bbox_ids_list)

    # 여기서 잘 모아주기
    rank, world_size = get_dist_info()
    device = torch.device(f'cuda:{rank}')

    # 각 bbox별로 전체 갯수 더하기
    num_total_data = [None for _ in range(world_size)]
    data_len = [len(loss_cls_tensor)]
    all_gather_object(num_total_data, data_len)
    max_num = max(num_total_data)  
    
    total_bbox_ids_tensor = total_tensor(bbox_ids_tensor, max_num)
    total_loss_cls_tensor = total_tensor(loss_cls_tensor, max_num)
    total_loss_bbox_tensor = total_tensor(loss_bbox_tensor, max_num)
    total_logits_tensor = total_tensor(logits_tensor, max_num)
    total_gt_labels_tensor = total_tensor(gt_labels_tensor, max_num)
    return total_bbox_ids_tensor, total_loss_cls_tensor, total_loss_bbox_tensor, total_logits_tensor, total_gt_labels_tensor


def multi_gpu_test(model, data_loader, tmpdir=None, gmm=None, gpu_collect=False, return_loss=False):    # 그냥 eval할때라.. 얘는 원래 함수 그대로 가져오거나 이하를 비슷하게 해야할 듯 
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, gmm=None, **data)
            # result = model(return_loss=return_loss, gmm=gmm, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results



def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results




# # Copyright (c) OpenMMLab. All rights reserved.
# import os.path as osp
# import pickle
# import shutil
# import tempfile
# import time

# import mmcv
# import torch
# import torch.distributed as dist
# from mmcv.image import tensor2imgs
# from mmcv.runner import get_dist_info

# from mmdet.core import encode_mask_results
# from typing import Optional, List

# import torch
# import torch.nn.functional as F


# def single_gpu_test(model,
#                     data_loader,
#                     show=False,
#                     out_dir=None,
#                     show_score_thr=0.3):
#     model.eval()
#     results = []
#     dataset = data_loader.dataset
#     PALETTE = getattr(dataset, 'PALETTE', None)
#     prog_bar = mmcv.ProgressBar(len(dataset))
#     for i, data in enumerate(data_loader):
#         with torch.no_grad():
#             result = model(return_loss=False, rescale=True, **data)

#         batch_size = len(result)
#         if show or out_dir:
#             if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
#                 img_tensor = data['img'][0]
#             else:
#                 img_tensor = data['img'][0].data[0]
#             img_metas = data['img_metas'][0].data[0]
#             imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
#             assert len(imgs) == len(img_metas)

#             for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
#                 h, w, _ = img_meta['img_shape']
#                 img_show = img[:h, :w, :]

#                 ori_h, ori_w = img_meta['ori_shape'][:-1]
#                 img_show = mmcv.imresize(img_show, (ori_w, ori_h))

#                 if out_dir:
#                     out_file = osp.join(out_dir, img_meta['ori_filename'])
#                 else:
#                     out_file = None

#                 model.module.show_result(
#                     img_show,
#                     result[i],
#                     bbox_color=PALETTE,
#                     text_color=PALETTE,
#                     mask_color=PALETTE,
#                     show=show,
#                     out_file=out_file,
#                     score_thr=show_score_thr)

#         # encode mask results
#         if isinstance(result[0], tuple):
#             result = [(bbox_results, encode_mask_results(mask_results))
#                       for bbox_results, mask_results in result]
#         # This logic is only used in panoptic segmentation test.
#         elif isinstance(result[0], dict) and 'ins_results' in result[0]:
#             for j in range(len(result)):
#                 bbox_results, mask_results = result[j]['ins_results']
#                 result[j]['ins_results'] = (bbox_results,
#                                             encode_mask_results(mask_results))

#         results.extend(result)

#         for _ in range(batch_size):
#             prog_bar.update()
#     return results


# def all_gather_into_tensor(output_tensor, input_tensor, group_size, group=None, async_op=False):
#     output_tensor = (
#         output_tensor
#         if not output_tensor.is_complex()
#         else torch.view_as_real(output_tensor)
#     )
#     input_tensor = (
#         input_tensor
#         if not input_tensor.is_complex()
#         else torch.view_as_real(input_tensor)
#     )
#     group = torch.distributed.new_group(list(range(group_size)))
#     work = group._allgather_base(output_tensor, input_tensor)

#     if async_op:
#         return work
#     else:
#         work.wait()




# def all_gather_object(object_list, obj, group=None):
#     import contextlib
#     import io
#     import logging
#     import os
#     import pickle
#     import time
#     import warnings
#     from datetime import timedelta
#     from typing import Callable, Dict, Optional, Tuple, Union
    
#     _pickler = pickle.Pickler
#     _unpickler = pickle.Unpickler
    
#     def _tensor_to_object(tensor, tensor_size):
#         buf = tensor.numpy().tobytes()[:tensor_size]
#         return _unpickler(io.BytesIO(buf)).load()

#     def _object_to_tensor(obj):
#         f = io.BytesIO()
#         _pickler(f).dump(obj)
#         byte_storage = torch.ByteStorage.from_buffer(f.getvalue())  # type: ignore[attr-defined]
#         # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
#         # Otherwise, it will casue 100X slowdown.
#         # See: https://github.com/pytorch/pytorch/issues/65696
#         byte_tensor = torch.ByteTensor(byte_storage)
#         local_size = torch.LongTensor([byte_tensor.numel()])
#         return byte_tensor, local_size


#     input_tensor, local_size = _object_to_tensor(obj)
#     current_device = torch.device("cuda", torch.cuda.current_device())
#     input_tensor = input_tensor.to(current_device)
#     local_size = local_size.to(current_device)

#     group_size = torch.distributed.get_world_size()
#     object_sizes_tensor = torch.zeros(
#         group_size, dtype=torch.long, device=current_device
#     )
#     object_size_list = [
#         object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
#     ]
#     group = torch.distributed.new_group(list(range(group_size)))
#     torch.distributed.all_gather(object_size_list, local_size, group=group)
#     max_object_size = int(max(object_size_list).item())  # type: ignore[type-var]
#     # Resize tensor to max size across all ranks.
#     input_tensor.resize_(max_object_size)
#     coalesced_output_tensor = torch.empty(
#         max_object_size * group_size, dtype=torch.uint8, device=current_device
#     )

#     output_tensors = [
#         coalesced_output_tensor[max_object_size * i : max_object_size * (i + 1)]
#         for i in range(group_size)
#     ]
#     torch.distributed.all_gather(output_tensors, input_tensor, group=group)

#     k = 0
#     for i, tensor in enumerate(output_tensors):
#         tensor = tensor.type(torch.uint8)
#         if tensor.device != torch.device("cpu"):
#             tensor = tensor.cpu()

#         tensor_size = object_size_list[i]
#         output = _tensor_to_object(tensor, tensor_size)
#         for j in range(len(output)):
#             object_list[k] = output[j]
#             k += 1    

# # added
# def all_gather(tensor: torch.Tensor, fixed_shape: Optional[List] = None) -> List[torch.Tensor]:
#     def compute_padding(shape, new_shape):
#         padding = []
#         for dim, new_dim in zip(shape, new_shape):
#             padding.insert(0, new_dim - dim)
#             padding.insert(0, 0)
#         return padding

#     input_shape = tensor.shape
#     if fixed_shape is not None:
#         padding = compute_padding(tensor.shape, fixed_shape)
#         if sum(padding) > 0:
#             tensor = F.pad(tensor, pad=padding, mode='constant', value=0)
    
#     output = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
#     dist.all_gather(output, tensor)

#     all_input_shapes = None
#     if fixed_shape is not None:
#         # gather all shapes
#         tensor_shape = torch.tensor(input_shape, device=tensor.device)
#         all_input_shapes = [torch.zeros_like(tensor_shape) for _ in range(dist.get_world_size())]
#         dist.all_gather(all_input_shapes, tensor_shape)
#         all_input_shapes = [t.tolist() for t in all_input_shapes]

#     if all_input_shapes:
#         for i, shape in enumerate(all_input_shapes):
#             padding = compute_padding(output[i].shape, shape)
#             if sum(padding) < 0:
#                 output[i] = F.pad(output[i], pad=padding)

#     return output

# def gmm_multi_gpu_test(model, data_loader, tmpdir=None, gmm=None, gpu_collect=False, return_loss=False):
#     """Test model with multiple gpus.

#     This method tests model with multiple gpus and collects the results
#     under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
#     it encodes results to gpu tensors and use gpu communication for results
#     collection. On cpu mode it saves the results on different gpus to 'tmpdir'
#     and collects them by the rank 0 worker.

#     Args:
#         model (nn.Module): Model to be tested.
#         data_loader (nn.Dataloader): Pytorch data loader.
#         tmpdir (str): Path of directory to save the temporary results from
#             different gpus under cpu mode.
#         gpu_collect (bool): Option to use either gpu or cpu to collect results.

#     Returns:
#         list: The prediction results.
#     """
#     model.eval()

#     bbox_ids_list = []
#     loss_cls_list = []
#     loss_bbox_list = []
#     logits_list = []
#     gt_labels_list = []

#     dataset = data_loader.dataset
    
#     # num_data = 36334  # 원래는 85xxx어쩌고였었음
#     num_bbox = len(dataset.coco.anns)
#     num_cls = len(dataset.coco.dataset['categories'])

#     rank, world_size = get_dist_info()
#     if rank == 0:
#         prog_bar = mmcv.ProgressBar(len(dataset))
#     time.sleep(2)  # This line can prevent deadlock problem in some cases.

#     for i, data in enumerate(data_loader):
#         with torch.no_grad():
#             box_ids, cls_loss, cls_labels, bbox_loss, logits = model(return_loss=return_loss, gmm=gmm, **data)     # TODO rescale loss 옵션을 받아서.. 


#         loss_cls_list.extend(cls_loss)
#         loss_bbox_list.extend(bbox_loss)
#         logits_list.extend(logits)
#         gt_labels_list.extend(cls_labels)
#         bbox_ids_list.extend(box_ids)

#         if rank == 0:
#             if gmm: batch_size = 1
#             else: batch_size = data['img'].size(0)
#             for _ in range(batch_size * world_size):
#                 prog_bar.update()
                
#     loss_cls_tensor = torch.stack(loss_cls_list)
#     loss_bbox_tensor = torch.stack(loss_bbox_list)
#     logits_tensor = torch.stack(logits_list)
#     gt_labels_tensor = torch.stack(gt_labels_list)
#     bbox_ids_tensor = torch.stack(bbox_ids_list)

#     '''
#     device = torch.device(f'cuda:{rank}')
#     # 각 bbox별로 전체 갯수 더하기
#     num_total_data = [None for _ in range(world_size)]
#     data_len = [len(bbox_ids_list)]
#     all_gather_object(num_total_data, data_len)
#     num_bbox = sum(num_total_data)  # total bbox len
#     max_num_bbox = max(num_total_data)  # total bbox len

#     # 한 rank에서의 tensor 값들 
#     loss_cls_list_tensor = torch.stack(loss_cls_list)
#     loss_bbox_list_tensor = torch.stack(loss_bbox_list)
#     logits_list_tensor = torch.stack(logits_list)
#     gt_labels_list_tensor = torch.stack(gt_labels_list)
#     bbox_ids_list_tensor = torch.stack(bbox_ids_list)

#     loss_cls_total = all_gather(loss_cls_list_tensor, fixed_shape=(max_num_bbox, ))[0] 
#     loss_bbox_total = all_gather(loss_bbox_list_tensor, fixed_shape=(max_num_bbox, ))[0] 
#     logits_total = all_gather(logits_list_tensor, fixed_shape=(max_num_bbox, num_cls))[0] 
#     gt_labels_total = all_gather(gt_labels_list_tensor, fixed_shape=(max_num_bbox, ))[0] 
#     bbox_ids_total = all_gather(bbox_ids_list_tensor, fixed_shape=(max_num_bbox, ))[0] 
#     # sample 갯수가 하나라.. 그냥 가져와도 괜찮을듯..
#     print(f'bbox_ids {bbox_ids_total[:10]}')
#     print(f'gt_labels_total {gt_labels_total[:10]}')
#     exit()
#     return bbox_ids_total, loss_cls_total, loss_bbox_total, logits_total, gt_labels_total
#     '''    
#     return bbox_ids_tensor, loss_cls_tensor, loss_bbox_tensor, logits_tensor, gt_labels_tensor


# def multi_gpu_test(model, data_loader, tmpdir=None, gmm=None, gpu_collect=False, return_loss=False):    # 그냥 eval할때라.. 얘는 원래 함수 그대로 가져오거나 이하를 비슷하게 해야할 듯 
#     """Test model with multiple gpus.

#     This method tests model with multiple gpus and collects the results
#     under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
#     it encodes results to gpu tensors and use gpu communication for results
#     collection. On cpu mode it saves the results on different gpus to 'tmpdir'
#     and collects them by the rank 0 worker.

#     Args:
#         model (nn.Module): Model to be tested.
#         data_loader (nn.Dataloader): Pytorch data loader.
#         tmpdir (str): Path of directory to save the temporary results from
#             different gpus under cpu mode.
#         gpu_collect (bool): Option to use either gpu or cpu to collect results.

#     Returns:
#         list: The prediction results.
#     """

#     model.eval()
#     results = []
#     dataset = data_loader.dataset
#     rank, world_size = get_dist_info()
#     if rank == 0:
#         prog_bar = mmcv.ProgressBar(len(dataset))
#     time.sleep(2)  # This line can prevent deadlock problem in some cases.
#     for i, data in enumerate(data_loader):
#         with torch.no_grad():
#             result = model(return_loss=False, rescale=True, gmm=None, **data)
#             # result = model(return_loss=return_loss, gmm=gmm, **data)
#             # encode mask results
#             if isinstance(result[0], tuple):
#                 result = [(bbox_results, encode_mask_results(mask_results))
#                           for bbox_results, mask_results in result]
#         results.extend(result)

#         if rank == 0:
#             batch_size = len(result)
#             for _ in range(batch_size * world_size):
#                 prog_bar.update()

#     # collect results from all ranks
#     if gpu_collect:
#         results = collect_results_gpu(results, len(dataset))
#     else:
#         results = collect_results_cpu(results, len(dataset), tmpdir)
#     return results



# def collect_results_cpu(result_part, size, tmpdir=None):
#     rank, world_size = get_dist_info()
#     # create a tmp dir if it is not specified
#     if tmpdir is None:
#         MAX_LEN = 512
#         # 32 is whitespace
#         dir_tensor = torch.full((MAX_LEN, ),
#                                 32,
#                                 dtype=torch.uint8,
#                                 device='cuda')
#         if rank == 0:
#             mmcv.mkdir_or_exist('.dist_test')
#             tmpdir = tempfile.mkdtemp(dir='.dist_test')
#             tmpdir = torch.tensor(
#                 bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
#             dir_tensor[:len(tmpdir)] = tmpdir
#         dist.broadcast(dir_tensor, 0)
#         tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
#     else:
#         mmcv.mkdir_or_exist(tmpdir)
#     # dump the part result to the dir
#     mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
#     dist.barrier()
#     # collect all parts
#     if rank != 0:
#         return None
#     else:
#         # load results of all parts from tmp dir
#         part_list = []
#         for i in range(world_size):
#             part_file = osp.join(tmpdir, f'part_{i}.pkl')
#             part_list.append(mmcv.load(part_file))
#         # sort the results
#         ordered_results = []
#         for res in zip(*part_list):
#             ordered_results.extend(list(res))
#         # the dataloader may pad some samples
#         ordered_results = ordered_results[:size]
#         # remove tmp dir
#         shutil.rmtree(tmpdir)
#         return ordered_results


# def collect_results_gpu(result_part, size):
#     rank, world_size = get_dist_info()
#     # dump result part to tensor with pickle
#     part_tensor = torch.tensor(
#         bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
#     # gather all result part tensor shape
#     shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
#     shape_list = [shape_tensor.clone() for _ in range(world_size)]
#     dist.all_gather(shape_list, shape_tensor)
#     # padding result part tensor to max length
#     shape_max = torch.tensor(shape_list).max()
#     part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
#     part_send[:shape_tensor[0]] = part_tensor
#     part_recv_list = [
#         part_tensor.new_zeros(shape_max) for _ in range(world_size)
#     ]
#     # gather all result part
#     dist.all_gather(part_recv_list, part_send)

#     if rank == 0:
#         part_list = []
#         for recv, shape in zip(part_recv_list, shape_list):
#             part_list.append(
#                 pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
#         # sort the results
#         ordered_results = []
#         for res in zip(*part_list):
#             ordered_results.extend(list(res))
#         # the dataloader may pad some samples
#         ordered_results = ordered_results[:size]
#         return ordered_results
