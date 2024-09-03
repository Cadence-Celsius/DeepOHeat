import torch, os, sys, matplotlib
import matplotlib.pyplot as plt
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append("../../")
from src import dataio_deeponet, modules, file_parser, dataio_utils
from src.utils import MyCmap

domain_0 = dict(
    domain_name=0,
    geometry=dict(
        starts=[0.0, 0.0, 0.0],
        ends=[1.0, 1.0, 0.5],
        num_intervals=[20, 20, 10],
        num_pde_points=2000,
        num_single_bc_points=200,
    ),
    conductivity_dist=dict(uneven_conductivity=False, background_conductivity=1),
    power=dict(
        bc=True,
        num_power_points_per_volume=2,
        num_power_points_per_surface=500,
        num_power_points_per_cell=5,
        power_map=dict(
            power_0=dict(
                type="surface_power",
                surface="top",
                location=dict(starts=(10, 0, 10), ends=(20, 10, 10)),
                params=dict(dim=2, value=1, weight=1),
            )
        ),
    ),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=-1, weight=1)),
    top=dict(bc=True),
    node=dict(root=True, leaf=True),
    parameterized=dict(variable=False),
)

domains_list = [domain_0]
global_params = {
    "loss_fun_type": "norm",
    "num_params_per_epoch": 50,
    "pde_params": dict(type="pde", params=dict(k=0.2, weight=1)),
}

print("Evaluating the DeepOHeat prototype: inference on given power map files")

device = "cuda:3"
model = modules.DeepONet(
    trunk_in_features=3,
    trunk_hidden_features=128,
    branch_in_features=441,
    branch_hidden_features=256,
    inner_prod_features=128,
    num_trunk_hidden_layers=3,
    num_branch_hidden_layers=7,
    nonlinearity="silu",
    freq=2 * torch.pi,
    std=1,
    freq_trainable=True,
    device=device,
)
print("The model used for this case:", model)

var = 1
len_scale = 0.3
print("var: {}, len scale: {}".format(var, len_scale))

dataset = dataio_deeponet.DeepONetMeshDataIO(
    domains_list, global_params, dim=2, var=var, len_scale=len_scale
)
mesh = dataio_utils.fixed_mesh_grid_3d(
    starts=[0.0, 0.0], ends=[1.0, 1.0], num_intervals=[20, 20]
)

root_path = "./log"
experiment_name = "experiment_1"
epoch = 10000
model_dir = os.path.join(
    root_path, experiment_name, "checkpoints", "model_epoch_{}.pth".format(epoch)
)
checkpoint = torch.load(model_dir, map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

figure_dir = os.path.join(
    root_path, experiment_name, "prototype_eval_new", "model_epoch_{}".format(epoch)
)
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

# Directory that keeps all the power map files that you want to evaluate
file_dir = "./power_maps"

eval_data = dataset.eval()
coords = eval_data["coords"]
eval_data = {key: value.float().to(device) for key, value in eval_data.items()}

cmap = MyCmap.get_cmap()
for file_name in os.listdir(file_dir):
    file_path = os.path.join(file_dir, file_name)
    sensor = file_parser.from_power_map_to_sensor(file_path)
    sensor = sensor.reshape(-1, order="F")

    fig = plt.figure()
    plt.scatter(mesh[:, 0], mesh[:, 1], c=sensor, cmap=cmap)
    plt.colorbar(ticks=np.linspace(sensor.min(), sensor.max(), 10))
    plt.savefig(os.path.join(figure_dir, file_name.split(".")[0] + "_map.png"))

    sensor = torch.tensor(
        sensor.reshape(1, -1).repeat(coords.shape[0], 0), device=device
    ).float()
    eval_data["beta"] = sensor

    u = model(eval_data)["model_out"].detach().cpu().numpy()
    u = 293.15 + 25 * u

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    sctt = ax.scatter3D(coords[:, 0], coords[:, 1], coords[:, 2], c=u, cmap=cmap)
    fig.colorbar(
        sctt, ax=ax, shrink=0.5, aspect=5, ticks=np.linspace(u.min(), u.max(), 10)
    )
    plt.savefig(os.path.join(figure_dir, file_name.split(".")[0] + "_results.png"))
    plt.close()
