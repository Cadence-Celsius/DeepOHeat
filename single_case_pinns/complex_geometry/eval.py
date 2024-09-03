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
        ends=[2.0, 1.0, 0.55],
        num_intervals=[20, 10, 11],
        num_pde_points=4000,
        num_single_bc_points=500,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=1),
    power=dict(
        bc=True,
        num_power_points_per_volume=4,
        num_power_points_per_surface=500,
        num_power_points_per_cell=5,
        power_map=dict(
            power_0=dict(
                type="volumetric_power",
                location=dict(starts=(0, 0, 5), ends=(20, 10, 6)),
                params=dict(k=0.2, value=1, weight=1),
            )
        ),
    ),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=-1, weight=1)),
    top=dict(bc=True, type="adiabatics", params=dict(dim=2, weight=1)),
    node=dict(root=True, leaf=False, children=dict(top=[1])),
    parameterized=dict(variable=False),
)

domain_1 = dict(
    domain_name=1,
    geometry=dict(
        starts=[0.0, 0.0, 0.55],
        ends=[1.0, 1.0, 0.75],
        num_intervals=[10, 10, 4],
        num_pde_points=800,
        num_single_bc_points=200,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=1),
    power=dict(bc=False),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=False),
    top=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=1, weight=1)),
    node=dict(root=False, leaf=True),
    parameterized=dict(variable=False),
)

domains_list = [domain_0, domain_1]
global_params = {
    "loss_fun_type": "norm",
    "num_params_per_epoch": 1,
    "pde_params": dict(type="pde", params=dict(k=0.2, weight=1)),
}

print(
    "Evaluating the trained single-case PINN: complex geometry and volumetric power defined in the middle layer"
)

device = "cuda:0"
model = modules.FFN(
    nonlinearity="silu",
    in_features=3,
    num_hidden_layers=3,
    hidden_features=128,
    device=device,
    freq=torch.pi,
    freq_trainable=True,
)
print("The model uesed for this case:", model)

root_path = "./log"
experiment_name = "experiment_1"
epoch = 5000
model_dir = os.path.join(
    root_path, experiment_name, "checkpoints", "model_epoch_{}.pth".format(epoch)
)
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint["model"])
model.eval()

figure_dir = os.path.join(root_path, experiment_name, "eval")
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

dataset = dataio.CuboidGeometryDataIO(domains_list, global_params)
eval_data = dataset.eval()
eval_data = {key: value.float().to(device) for key, value in eval_data.items()}

u = model(eval_data)["model_out"].detach().cpu().numpy().squeeze()
u = 293.15 + 25 * u

mesh = eval_data["coords"].detach().cpu().numpy().squeeze()

cmap = MyCmap.get_cmap()

fig = plt.figure()
ax = plt.axes(projection="3d")
sctt = ax.scatter3D(mesh[:, 0], mesh[:, 1], mesh[:, 2], c=u, cmap=cmap)
fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5, ticks=np.linspace(u.min(), u.max(), 10))
plt.savefig(os.path.join(figure_dir, "eval_epoch_{}.png".format(epoch)))
plt.close("all")
