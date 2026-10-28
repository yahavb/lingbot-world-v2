"""NEURON_RT_VISIBLE_CORES order that lets the job's TP8 x CP8 (LNC2, 64 ranks) run on this host.

    NEURON_RT_VISIBLE_CORES=$(python3 kiro/core_order.py) torchrun --nproc_per_node=64 ...

torch-neuronx gives local rank i the i-th entry of a comma-separated NEURON_RT_VISIBLE_CORES, so
the list sets which device each rank lands on. The 16 devices are a 4 x 4 torus (device d =
row d // 4, column d % 4; neighbors one step along a row or column, wrapping). With the default
order, TP8 pairs devices (0,1), (2,3), ... and each CP8 group is every other device -- columns 0
and 2, two rings with no link between them -- so the collectives library cannot build the group
("Failed to find device to device paths"). This order pairs linked devices and makes each CP8
group the devices of two adjacent columns, which form one ring: CP groups {0,3,7,4,8,11,15,12}
and {1,2,6,5,9,10,14,13}. Group ranks stay contiguous / strided exactly as upstream builds them.
"""
ORDER = [0, 1, 3, 2, 7, 6, 4, 5, 8, 9, 11, 10, 15, 14, 12, 13]


def neighbors(d):
    r, c = divmod(d, 4)
    return {4 * r + (c + 1) % 4, 4 * r + (c - 1) % 4, 4 * ((r + 1) % 4) + c, 4 * ((r - 1) % 4) + c}


def check(order):
    pairs = [order[i:i + 2] for i in range(0, len(order), 2)]
    assert all(b in neighbors(a) for a, b in pairs), "a TP pair is not directly linked"
    for g in ([p[0] for p in pairs], [p[1] for p in pairs]):
        assert all(g[(i + 1) % len(g)] in neighbors(g[i]) for i in range(len(g))), f"CP group {g} not a ring"


if __name__ == "__main__":
    check(ORDER)
    print(",".join(str(4 * d + c) for d in ORDER for c in range(4)))
