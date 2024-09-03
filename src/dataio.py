import torch
import copy
from torch.utils.data import Dataset
from src.geometry import create_stacking_cuboidal_geometry, fetch_data


class CuboidGeometryDataIO(Dataset):

    def __init__(self, domains_list, global_params, beta_as_input=False):
        super().__init__()
        self.geometry = create_stacking_cuboidal_geometry(domains_list)
        self.pde_params = global_params["pde_params"]
        self.loss_fun_type = global_params["loss_fun_type"]
        self.num_params_per_epoch = global_params["num_params_per_epoch"]

        self.beta_as_input = beta_as_input
        self.mode = "train"
        self.args = []

    def __len__(self):
        return self.num_params_per_epoch

    def __getitem__(self, idx):
        return fetch_data(self.geometry, self.mode, *self.args)

    def train(self, sample_domain=True):
        self.mode = "train"
        self.args = [sample_domain]
        coords, conductivity, beta = next(iter(self))
        beta = beta.repeat(coords.shape[0], 1)

        if self.beta_as_input:
            coords = torch.concat([coords, beta], 1)

        return (
            {"coords": coords, "beta": beta},
            conductivity,
            copy.deepcopy(self.geometry),
        )

    def eval(self, sample_domain=True, sample_mode="middle", res=25):
        self.mode = "eval"
        self.args = [sample_domain, sample_mode, res]
        coords, _, beta = next(iter(self))
        beta = beta.repeat(coords.shape[0], 1)

        if self.beta_as_input:
            coords = torch.concat([coords, beta], 1)

        return {"coords": coords, "beta": beta}
