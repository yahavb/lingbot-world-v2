"""Exact latent grids, in a module with NO imports.

Separate from wan/configs/__init__.py on purpose: the job preflight needs this table to
validate the patch grid, and it runs BEFORE the dependency install. Importing
wan.configs there pulls in wan_i2v_A14B -> easydict and died with ModuleNotFoundError,
after the preflight had already aborted one run by computing the aspect-fitted grid (78)
instead of the forced one (72).

Load it by file path, no package import:
    importlib.util.spec_from_file_location("eg", "wan/configs/exact_grids.py")
"""
# EXACT LATENT GRIDS, bypassing the aspect fit.
#
# Every other preset is an AREA target: lat_h/lat_w come from sqrt(area * aspect) so the
# grid follows the input image. That cannot hit an exact shape, and the reference implementation's const-max
# AG attention requires one -- it asserts the KV chunk is a whole number of 128-token
# tiles (reference const_max_ag.py:28), ATTENTION_TILE_SIZE). With examples/03 (5145x2894,
# 16:9) the 768*1152 area target fits to lat 88x156, giving frame_seqlen 3432 which is
# 104 short of a tile boundary, and patch_w 78 which is not divisible by cp=4 either.
#
# Forcing lat 96x144 gives frame_seqlen 3456 = 27 tiles exactly, patch grid 48x72 with
# patch_w % 4 == 0 for cp4, and lat_w 144 which the VAE width padding takes to 160 so all
# 32 ranks decode. It also STRETCHES a 16:9 source to 2:3, which is the deliberate cost of
# matching the reference implementation's shape.
#
EXACT_GRID_SIZES = {
    '768*1152': (768, 1152),
    # 480*768 -> 30h x 48w, L = 1440: the only 480p grid that runs at cp8. Both
    # L % 8 == 0 and 48 % 8 == 0 hold, so flat and width sharding give the same local
    # length. (480*832 is deliberately NOT an exact grid: forced to 30x52 it shards
    # flat but fails at cp8 on a width-derived length, and at cp4 the aspect-fitted
    # 29x52 grid, L = 1508, is the one that was measured.)
    '480*768': (480, 768),
}
