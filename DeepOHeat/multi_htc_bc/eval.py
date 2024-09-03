import torch, os, sys, matplotlib
import matplotlib.pyplot as plt
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append("../../")
from src import dataio, modules
from src.utils import MyCmap

domain_0 = dict(
    domain_name=0,
    geometry=dict(
        starts=[0.0, 0.0, 0.0],
        ends=[1.0, 1.0, 0.55],
        num_intervals=[20, 20, 11],
        num_pde_points=4000,
        num_single_bc_points=500,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=1),
    power=dict(
        bc=True,
        num_power_points_per_volume=2,
        num_power_points_per_surface=500,
        num_power_points_per_cell=5,
        power_map=dict(
            power_0=dict(
                type="volumetric_power",
                location=dict(starts=(0, 0, 5), ends=(20, 20, 6)),
                params=dict(k=0.2, value=1, weight=1),
            )
        ),
    ),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=-1, weight=1)),
    top=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=1, weight=1)),
    node=dict(root=True, leaf=True),
    parameterized=dict(
        variable=True,
        param_space=dict(
            top=dict(k={"param_range": (0.1, 0.3), "type": "continuous"}),
            bottom=dict(k={"param_range": (0.1, 0.3), "type": "continuous"}),
        ),
    ),
)

domains_list = [domain_0]
global_params = {
    "loss_fun_type": "norm",
    "num_params_per_epoch": 20,
    "pde_params": dict(type="pde", params=dict(k=0.2, weight=1)),
}

print(
    "Evaluating trained DeepOHeat: parameterized top HTC BC with volumetric power defined in the middle layer"
)

device = "cuda:2"
model = modules.MIONet(
    trunk_in_features=3,
    trunk_hidden_features=128,
    branch_in_features=2,
    branch_hidden_features=20,
    inner_prod_features=50,
    num_hidden_layers=3,
    nonlinearity="silu",
    freq=torch.pi,
    std=1,
    freq_trainable=True,
    device=device,
)
print("The model used for this case:", model)

root_path = "./log"
experiment_name = "experiment_1"
epoch = 5000
model_dir = os.path.join(
    root_path, experiment_name, "checkpoints", "model_epoch_{}.pth".format(epoch)
)
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint["model"])
model.eval()

figure_dir = os.path.join(
    root_path, experiment_name, "eval", "model_epoch_{}".format(epoch)
)
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

dataset = dataio.CuboidGeometryDataIO(domains_list, global_params)
cmap = MyCmap.get_cmap()

for sample_mode in ["low", "middle", "high"]:
    eval_data = dataset.eval(sample_mode=sample_mode)
    eval_data = {key: value.float().to(device) for key, value in eval_data.items()}

    u = model(eval_data)["model_out"].detach().cpu().numpy().squeeze()
    u = 293.15 + 25 * u

    mesh = eval_data["coords"].detach().cpu().numpy().squeeze()
    beta = eval_data["beta"][0, :].detach().cpu().numpy()

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    sctt = ax.scatter3D(mesh[:, 0], mesh[:, 1], mesh[:, 2], c=u, cmap=cmap)
    fig.colorbar(
        sctt, ax=ax, shrink=0.5, aspect=5, ticks=np.linspace(u.min(), u.max(), 10)
    )
    plt.savefig(os.path.join(figure_dir, "eval_beta_{}.png".format(beta)))

plt.close("all")
