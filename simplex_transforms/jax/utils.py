import jax


def vmap_over_leading_axes(fun, arr):
    num_leading_axes = arr.ndim - 1
    vmapped_fun = fun
    for _ in range(num_leading_axes):
        vmapped_fun = jax.vmap(vmapped_fun)
    return vmapped_fun(arr)
