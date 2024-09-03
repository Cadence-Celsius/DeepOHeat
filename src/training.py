import shutil
import torch
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from src.diff_operator import laplacian_jacobian
from src.utils import MyCmap


def val_fn_init(half_geometry=False, slice_dim=None, slice_value=None):
    if half_geometry and slice_dim is None and slice_value is None:
        raise ValueError(f"No slice coordinate is given")
    cmap = MyCmap.get_cmap()

    def val_fn(dataset, model, epoch, figure_dir, device=None):

        eval_data = dataset.eval()

        if device is not None:
            eval_data = {key: value.to(device) for key, value in eval_data.items()}

        eval_data = {key: value.float() for key, value in eval_data.items()}
        model.eval()

        u = model(eval_data)["model_out"].detach().cpu().numpy().squeeze()
        u = 293.15 + 25 * u

        mesh = eval_data["coords"].detach().cpu().numpy().squeeze()

        if half_geometry:
            slice_idx = mesh[:, slice_dim] >= slice_value
            mesh = mesh[slice_idx, :]
            u = u[slice_idx]

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        sctt = ax.scatter3D(mesh[:, 0], mesh[:, 1], mesh[:, 2], c=u, cmap=cmap)
        fig.colorbar(
            sctt, ax=ax, shrink=0.5, aspect=5, ticks=np.linspace(u.min(), u.max(), 10)
        )
        plt.savefig(os.path.join(figure_dir, "pred_epoch_{}.png".format(epoch)))
        plt.close("all")

    return val_fn


def train(
    model,
    dataset,
    epochs,
    lr,
    epochs_til_checkpoints,
    model_dir,
    loss_fn,
    val_fn=None,
    device=None,
    start_epoch=0,
    lr_decay=False,
    epochs_til_decay=100,
    epochs_til_val=500,
    deeponet=False,
):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    lr_decayed = lr

    if start_epoch > 0:
        model_path = os.path.join(
            model_dir, "checkpoints", "model_epoch_{}.pth".format(start_epoch)
        )
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model"])
        model.train()
        optim.load_state_dict(checkpoint["optim"])
        optim.param_groups[0]["lr"] = lr
        assert start_epoch == checkpoint["epoch"]

    else:
        if os.path.exists(model_dir):
            val = input("Path %s already exists, overwrite? (y/n)" % model_dir)
            if val == "y":
                shutil.rmtree(model_dir)
        os.makedirs(model_dir)

    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    figure_dir = os.path.join(model_dir, "figure")
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    def generate_loss_dicts():
        loss_dict = {
            "pde": 0,
            "htc": 0,
            "adiabatics": 0,
            "volumetric_power": 0,
            "surface_power": 0,
            "neumann": 0,
            "dirichelet": 0,
        }
        loss_idx_count_dict = loss_dict.copy()

        return loss_dict, loss_idx_count_dict

    total_loss_list = []
    epoch = 0
    # dataset.draw_power_map(model_dir)
    for epoch in range(start_epoch, epochs):
        if val_fn and not epoch % epochs_til_val and epoch:
            val_fn(dataset, model, epoch, figure_dir, device)

        if not epoch % epochs_til_checkpoints and epoch:
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optim.state_dict(),
            }
            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, "model_epoch_{}.pth".format(epoch)),
            )

        if not epoch % epochs_til_decay and epoch and lr_decay:
            lr_decayed = lr_decayed * 0.9
            optim = torch.optim.Adam(lr=lr_decayed, params=model.parameters())
            print("lr decay, current: {}".format(lr_decayed))

        loss_dict, _ = generate_loss_dicts()
        starting_idx = 0
        coords_list, beta_list = [], []
        conductivity_list, geometry_list, idx_list = [], [], []

        for step in range(len(dataset)):
            model_input, conductivity, geometry_step = dataset.train()

            if device is not None:
                model_input = {
                    key: value.to(device).float() for key, value in model_input.items()
                }
                conductivity = conductivity.to(device).float()

            # model_input = {key:value.float() for key, value in model_input.items()}
            coords_list.append(model_input["coords"])
            beta_list.append(model_input["beta"])

            ending_idx = int(starting_idx + (model_input["coords"]).shape[0])
            idx_list.append((starting_idx, ending_idx))
            starting_idx = ending_idx

            conductivity_list.append(conductivity)
            geometry_list.append(geometry_step)

        model_input = dict(
            coords=torch.concat(coords_list, 0), beta=torch.concat(beta_list, 0)
        )
        model_output = model(model_input)

        u = model_output["model_out"]
        coords = model_output["model_in"]

        conductivity = torch.concat(conductivity_list, 0)

        jac, u_laplace = laplacian_jacobian(u, coords, conductivity)

        for step in range(len(dataset)):
            idx_step = idx_list[step]
            start_idx, end_idx = idx_step[0], idx_step[1]

            geometry_step = geometry_list[step]
            loss_dict_step = loss_fn(
                u[start_idx:end_idx, ...],
                jac[start_idx:end_idx, ...],
                u_laplace[start_idx:end_idx, ...],
                geometry_step,
            )

            if loss_dict.keys() != loss_dict_step.keys():
                raise ValueError(
                    f"Loss dict in current step is not aligned with the pre-defined loss dict, unable to merge"
                )
            else:
                for key in loss_dict.keys():
                    loss_dict[key] += loss_dict_step[key]

        loss_dict["total_loss"] = sum([*loss_dict.values()]) / len(dataset)

        print("epoch: %d" % epoch, loss_dict)

        optim.zero_grad()
        loss_dict["total_loss"].backward()
        optim.step()

        total_loss_list.append(loss_dict["total_loss"].item())

    plt.figure(figsize=(16, 8))
    plt.plot(np.arange(len(total_loss_list)), total_loss_list)
    plt.yscale("log")
    plt.savefig(os.path.join(figure_dir, "loss.png"))
    plt.close("all")

    val_fn(dataset, model, epoch + 1, figure_dir, device)

    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
    }
    torch.save(
        checkpoint, os.path.join(checkpoint_dir, "model_epoch_{}.pth".format(epoch + 1))
    )
