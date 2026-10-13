import torch
import torch.distributed as dist


def init_distributed_group():
    """r initialize sequence parallel group.
    """
    if not dist.is_initialized():
        dist.init_process_group(backend='neuron')


def get_rank():
    return dist.get_rank()


def get_world_size():
    return dist.get_world_size()


def all_to_all(x, scatter_dim, gather_dim, group=None, world_size=None, **kwargs):
    """
    `scatter` along one dimension and `gather` along another.

    `world_size` may be passed explicitly (e.g. an SP sub-group's size) when
    `group` is not the default world group; otherwise the size is read back
    from `group` (or the world) so chunking always matches the group that the
    collective runs over.
    """
    if world_size is None:
        world_size = dist.get_world_size(group) if group is not None else get_world_size()
    if world_size > 1:
        inputs = [u.contiguous() for u in x.chunk(world_size, dim=scatter_dim)]
        outputs = [torch.empty_like(u) for u in inputs]
        dist.all_to_all(outputs, inputs, group=group, **kwargs)
        x = torch.cat(outputs, dim=gather_dim).contiguous()
    return x


def all_gather(tensor, group=None, world_size=None):
    """all_gather over `group` (default: world). `world_size` may be passed to
    match a sub-group's size; reads it back from `group` when omitted."""
    if world_size is None:
        world_size = dist.get_world_size() if group is None else dist.get_world_size(group)
    if world_size == 1:
        return [tensor]
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor, group=group)
    return tensor_list


def gather_forward(input, dim, group=None, world_size=None):
    """Gather a sequence-chunked tensor back to full length over `group`."""
    if world_size is None:
        world_size = dist.get_world_size() if group is None else dist.get_world_size(group)
    if world_size == 1:
        return input
    output = all_gather(input, group=group, world_size=world_size)
    return torch.cat(output, dim=dim).contiguous()
