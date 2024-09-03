from numpy import gradient
import torch
from torch.autograd import grad


def gradients(y, x, create_graph=True, retain_graph=True):
    return grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=retain_graph,
    )


# def jacobian(y, x):
#     num_obs = y.shape[0]
#     jac = torch.zeros(num_obs, y.shape[-1], x.shape[-1]).to(y.device)
#     for i in range(y.shape[-1]):
#         y_flat = y[..., i].view(-1, 1)
#         jac[:, i, :] = grad(y_flat, x, grad_outputs=torch.ones_like(y_flat), create_graph=True, retain_graph=True)[0]

#     return jac


def jacobian(y, x):
    jac = gradients(y, x)[0]
    return jac


def laplacian(u, coords):
    jac = jacobian(u, coords)
    ux, uy, uz = jac[..., 0, 0], jac[..., 0, 1], jac[..., 0, 2]
    uxx, uyy, uzz = (
        gradients(ux, coords)[0][..., 0],
        gradients(uy, coords)[0][..., 1],
        gradients(uz, coords)[0][..., 2],
    )

    return ux, uy, uz, uxx + uyy + uzz


def laplacian_with_conductivity(u, coords, conductivity):
    jac = jacobian(u, coords)
    ux, uy, uz = jac[..., 0, 0], jac[..., 0, 1], jac[..., 0, 2]
    uxx, uyy, uzz = (
        gradients(ux * conductivity, coords)[0][..., 0],
        gradients(uy * conductivity, coords)[0][..., 1],
        gradients(uz * conductivity, coords)[0][..., 2],
    )

    return ux, uy, uz, uxx + uyy + uzz


# def laplacian_jacobian(u, coords, conductivity):
#     jac = jacobian(u, coords)
#     ux, uy, uz = jac[..., 0, 0], jac[..., 0, 1], jac[..., 0, 2]
#     uxx, uyy, uzz = gradients(ux*conductivity, coords)[0][..., 0], \
#         gradients(uy*conductivity, coords)[0][..., 1], gradients(uz*conductivity, coords)[0][..., 2]

#     return jac, uxx+uyy+uzz

# def laplacian_jacobian(u, coords, conductivity):
#     jac = jacobian(u, coords)
#     jac_cond = jac * (conductivity.reshape(-1, 1))
#     jac_second = gradients(jac_cond, coords)[0]

#     return jac, jac_second[:, 0]+jac_second[:, 1]+jac_second[:, 2]


def laplacian_jacobian(u, coords, conductivity):
    jac = jacobian(u, coords)
    uxx, uyy, uzz = (
        gradients(jac[..., 0] * conductivity, coords)[0][..., 0],
        gradients(jac[..., 1] * conductivity, coords)[0][..., 1],
        gradients(jac[..., 2] * conductivity, coords)[0][..., 2],
    )
    # jac_cond = jac * (conductivity.reshape(-1, 1))

    # jac_second = gradients(jac_cond, coords)[0]

    return jac, uxx + uyy + uzz
    # return jac, jac_second[:, 0]+jac_second[:, 1]+jac_second[:, 2]
