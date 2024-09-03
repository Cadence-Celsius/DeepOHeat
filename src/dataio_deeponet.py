import torch, copy, os
import numpy as np
from torch.utils.data import Dataset

from src.dataio_utils import fixed_mesh_grid_3d
from src.geometry_deeponet import (
    create_stacking_cuboidal_geometry,
    fetch_mesh_data,
)
import matplotlib.pyplot as plt

# The dataio classes used in training DeepOHeat: 2D power map


class DeepONetMeshDataIO(Dataset):

    def __init__(self, domains_list, global_params, dim=2, var=1, len_scale=0.3):
        super().__init__()
        self.geometry = create_stacking_cuboidal_geometry(
            domains_list, dim=dim, mesh=True
        )
        self.pde_params = global_params["pde_params"]
        self.loss_fun_type = global_params["loss_fun_type"]
        self.num_params_per_epoch = global_params["num_params_per_epoch"]

        self.dim = dim
        self.mode = "train"
        self.var = var
        self.len_scale = len_scale

    def __len__(self):
        return self.num_params_per_epoch

    def __getitem__(self, idx):
        return fetch_mesh_data(
            self.geometry,
            self.mode,
            dim=self.dim,
            var=self.var,
            len_scale=self.len_scale,
        )

    def draw_power_map(self, model_dir):
        train_data, _, _ = self.train()
        train_coords, train_sensors = train_data.values()
        train_sensor_power = train_sensors[0, :].reshape(-1)
        starts, ends, num_intervals, _, _ = self.geometry.domain["geometry"].values()
        mesh = fixed_mesh_grid_3d(
            starts=starts[:2], ends=ends[:2], num_intervals=num_intervals[:2]
        )

        fig_dir = os.path.join(model_dir, "figure")

        fig = plt.figure()
        plt.scatter(mesh[:, 0], mesh[:, 1], c=train_sensor_power, cmap="jet")
        plt.colorbar(
            ticks=np.linspace(train_sensor_power.min(), train_sensor_power.max(), 10)
        )
        plt.savefig(os.path.join(fig_dir, "train_sensor.png"))
        plt.close("all")

        self.geometry.update_set()
        top_idx_set = self.geometry.boundaries_set["top"]
        top_coords = train_coords[top_idx_set, :]

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        sctt = ax.scatter3D(
            top_coords[:, 0],
            top_coords[:, 1],
            top_coords[:, 2],
            c=train_sensor_power,
            cmap="jet",
        )
        fig.colorbar(
            sctt,
            ax=ax,
            shrink=0.5,
            aspect=5,
            ticks=np.linspace(train_sensor_power.min(), train_sensor_power.max(), 10),
        )
        plt.savefig(os.path.join(fig_dir, "coords_power_map.png"))

        eval_data = self.eval()
        _, power_map = eval_data.values()
        power_map = power_map[0, :].reshape(-1)

        fig = plt.figure()
        plt.scatter(mesh[:, 0], mesh[:, 1], c=power_map, cmap="jet")
        plt.colorbar(ticks=np.linspace(power_map.min(), power_map.max(), 10))
        plt.savefig(os.path.join(fig_dir, "eval_power_map.png"))
        plt.close("all")

    def train(self):
        self.mode = "train"
        sensors, coords, conductivity = next(iter(self))
        sensors = sensors.reshape(1, -1).repeat(coords.shape[0], 0)
        return (
            {"coords": torch.tensor(coords), "beta": torch.tensor(sensors)},
            torch.tensor(conductivity),
            copy.deepcopy(self.geometry),
        )

    def eval(self):
        self.mode = "eval"
        sensors, coords, _ = next(iter(self))
        sensors = sensors.reshape(1, -1).repeat(coords.shape[0], 0)

        return {"coords": torch.tensor(coords), "beta": torch.tensor(sensors)}
