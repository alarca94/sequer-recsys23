import torch


def get_span_ixs(start_ixs, span_size):
    return start_ixs + torch.arange(span_size).repeat(start_ixs.shape[0], 1)


def gather_span(data, start_ixs, span_size, dim=1):
    select_ixs = get_span_ixs(start_ixs, span_size)
    return torch.gather(data, index=select_ixs, dim=dim)
