import json
import random
import math
import torch.distributed as dist
from typing import Tuple, List
from dataclasses import dataclass
from copy import deepcopy


def safe_get_rank() -> int:
    try:
        rank = dist.get_rank()
    except:
        rank = 0
    return rank


@dataclass
class ApolloInfo:
    curr_meta_layer: int
    extend_layer_list: list[int]
    do_stack: bool = False
    stack_info: list[int] = None


def extend_meta_layer(src: int, tgt: int) -> List[int]:
    ratio = tgt / src
    return [int(tgt_idx / ratio) for tgt_idx in range(tgt)]


def stack_meta_layer(src: int, tgt: int) -> List[int]:
    def do_rev(tgt_l: List[int]) -> List[int]:
        # print([list(reversed(tgt_l[idx:idx+src])) for idx in range(0, tgt, src)])
        res = sum(
            [list(reversed(tgt_l[idx : idx + src])) for idx in range(0, tgt, src)], []
        )
        return res

    return do_rev([((-tgt_idx - 1) % src) for tgt_idx in range(tgt)])


def get_curr_apollo_info(
    unique_layer: int, max_layer: int, dist_function: str, grow_method: str
) -> ApolloInfo:
    if dist_function == "uniform":
        extend_layer_num = random.choice(list(range(unique_layer, max_layer + 1)))
    elif dist_function == "lvps":
        x_min = 1e-6
        ratio = max_layer / unique_layer
        begin = ratio + 0.4
        end = 0.6
        n = (begin * x_min - end) / (end - begin)
        m = end * (n + 1)
        x = random.uniform(x_min, 1.0)
        y = m / (x + n)
        y = min(y, ratio)
        y = max(y, 1)
        extend_layer_num = int(y * unique_layer)
    elif dist_function == "sigmoid":
        x_min = 1e-6
        ratio = max_layer / unique_layer
        begin = ratio + 0.4
        end = 0.6
        x = random.uniform(x_min, 1.0)
        y = (begin - end) / (1 + math.exp(20 * (x - 0.5))) + end
        y = min(y, ratio)
        y = max(y, 1)
        extend_layer_num = int(y * unique_layer)
    elif dist_function == "constant_high":
        extend_layer_num = max_layer
    else:
        raise NotImplementedError(
            "Fuction {} is not implemented.".format(dist_function)
        )

    if grow_method == "extend":
        extend_layer_list = extend_meta_layer(unique_layer, extend_layer_num)
    elif grow_method == "stack":
        extend_layer_list = stack_meta_layer(unique_layer, extend_layer_num)
    else:
        raise ValueError("The grow method %s is not supported." % grow_method)
    return ApolloInfo(extend_layer_num, extend_layer_list)


if __name__ == "__main__":
    print(get_curr_apollo_info(3, 12, "lvps", "extend"))
