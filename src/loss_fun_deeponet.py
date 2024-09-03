import torch
import numpy as np
from ordered_set import OrderedSet
from src.diff_operator import *
import matplotlib.pyplot as plt
from src.geometry_utils import iterate_over_entire_geometry


def cal_vec_loss(loss_fun_type, vec, weight=1):
    if loss_fun_type == "mse":
        loss = torch.nn.functional.mse_loss(vec, torch.zeros_like(vec))
    elif loss_fun_type == "norm":
        loss = vec.norm()
    elif loss_fun_type == "squared_norm":
        loss = vec.norm() ** 2
    elif loss_fun_type == "msn":
        loss = vec.norm() ** 2 / len(vec)

    return loss * weight if not torch.isnan(loss).item() else 0


def loss_adiabatics(loss_fun_type, u, jac, u_laplace, idx, dim, weight=1):
    # vec = jac[..., 0, dim][idx].squeeze()
    vec = jac[..., dim][idx].squeeze()
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_dirichelet(loss_fun_type, u, jac, u_laplace, idx, value, weight=1):
    vec = (u[idx, :] - torch.ones_like(u[idx, :]) * value).squeeze()
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_robin(loss_fun_type, u, jac, u_laplace, idx, dim, k, direction, weight=1):
    # vec = (u[idx, :]-0.2+direction*k*jac[..., 0, dim][idx]).squeeze()
    vec = (u[idx, :].squeeze() - 0.2 + direction * k * jac[..., dim][idx]).squeeze()
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_pde(loss_fun_type, u, jac, u_laplace, idx, k=0.2, weight=1):
    vec = (u_laplace[idx] * k).squeeze()
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_volumetric_power(
    loss_fun_type, u, jac, u_laplace, idx, k=0.2, value=1, weight=1
):
    vec = (u_laplace[idx] * k + value * torch.ones_like(u_laplace[idx])).squeeze()
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_neumann(loss_fun_type, u, jac, u_laplace, idx, dim=2, weight=1):
    # vec = jac[..., 0, dim][idx].squeeze()
    vec = jac[..., dim][idx].squeeze()
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_surface_power(loss_fun_type, u, jac, u_laplace, idx, dim, value, weight=1):
    # vec = (jac[..., 0, dim][idx] - torch.ones_like(jac[..., 0, dim][idx])*value).squeeze()
    vec = (jac[..., dim][idx] - torch.ones_like(jac[..., dim][idx]) * value).squeeze()
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_arbitrary_surface_power(loss_fun_type, jac, q, idx, weight=1):
    # vec = (jac[..., 0, dim][idx] - torch.ones_like(jac[..., 0, dim][idx])*value).squeeze()
    vec = jac[..., 2][idx].squeeze() - q[idx]
    return cal_vec_loss(loss_fun_type, vec, weight)


def loss_mesh_arbitrary_surface_power(loss_fun_type, jac, q, idx, weight=1):
    # vec = (jac[..., 0, dim][idx] - torch.ones_like(jac[..., 0, dim][idx])*value).squeeze()
    vec = jac[..., 2][idx].squeeze() - q
    return cal_vec_loss(loss_fun_type, vec, weight)


def find_boundaries_endpoints(starts, ends):
    boundaries_dict = dict(
        front=dict(starts=starts, ends=[ends[0], starts[1], ends[2]]),
        back=dict(starts=[starts[0], ends[1], starts[2]], ends=ends),
        left=dict(starts=starts, ends=[starts[0], ends[1], ends[2]]),
        right=dict(starts=[ends[0], starts[1], starts[2]], ends=ends),
        bottom=dict(starts=starts, ends=[ends[0], ends[1], starts[2]]),
        top=dict(starts=[starts[0], starts[1], ends[2]], ends=ends),
    )

    return boundaries_dict


def mesh_loss_fun_geometry_init(dataset):
    pde_params = dataset.pde_params
    loss_fun_type = dataset.loss_fun_type

    bc_loss_fun_dict = {
        "pde": loss_pde,
        "htc": loss_robin,
        "adiabatics": loss_adiabatics,
        "volumetric_power": loss_volumetric_power,
        "surface_power": loss_surface_power,
        "neumann": loss_neumann,
        "dirichelet": loss_dirichelet,
    }

    def top_2d_power_loss_fn(u, jac, u_laplace, beta, geometry):

        loss_dict = {
            "pde": 0,
            "htc": 0,
            "adiabatics": 0,
            "volumetric_power": 0,
            "surface_power": 0,
            "neumann": 0,
            "dirichelet": 0,
        }
        pde_set_list = []

        def bc_loss_cal(boundary_dict, boundary_idx):
            loss_type = boundary_dict["type"]
            loss_fun = bc_loss_fun_dict[loss_type]
            loss_dict[loss_type] += loss_fun(
                loss_fun_type,
                u,
                jac,
                u_laplace,
                boundary_idx,
                *boundary_dict["params"].values(),
            )

        def single_node_loss_fun(node):
            node.update_set()
            pde_set_list.append(node.pde_set)

            for boundary_name, boundary_set in node.boundaries_set.items():
                boundary_dict = node.domain_step[boundary_name]

                if boundary_name == "top":
                    power_map = beta[0, :].reshape(-1)
                    loss_dict["surface_power"] += loss_mesh_arbitrary_surface_power(
                        loss_fun_type, jac, power_map, boundary_set
                    )
                    continue

                if not boundary_dict["bc"]:
                    continue

                bc_loss_cal(boundary_dict, boundary_set)

        iterate_over_entire_geometry(geometry, single_node_loss_fun)
        pde_set = OrderedSet().union(*pde_set_list)
        bc_loss_cal(pde_params, list(pde_set))

        return loss_dict

    return top_2d_power_loss_fn
