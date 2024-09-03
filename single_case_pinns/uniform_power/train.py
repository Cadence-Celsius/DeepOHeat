import time, torch, os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append("../../")
from src import dataio, training, loss_fun, modules

# make sure the root path is appended into sys else won't able to import src

# define domain
# domain_name: current not functioning
# geometry: define the starts and ends of each axis, num_intervals corresponds to the resolution of FEM, which defines the smallest power size
# conductivity: if not uneven_conductivity then uniformly equal to background_conductivity, else needs to define seperately
# power: bc indicates if this domain has power, if True, define all the powers in power_map.
#        num_power_points_per_volume, num_power_points_per_surface, num_power_points_per_cell define how many extra points should be sampled inside each power
# front, back, ...: correspond to each boundary of the cuboid. bc indicates if it satisfies certain BC, if True, define type and params (see details in loss_fun.py)
# node: if single domain then root==True, leaf==True; if it has child node, then leaf==False, and define its child; if it has parent node, then root==False
# parameterized: variable=False for single-case PINNs, True for single/multiple variable DeepOHeat
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
        num_power_points_per_volume=5,
        num_power_points_per_surface=500,
        num_power_points_per_cell=5,
        power_map=dict(
            power_0=dict(
                type="surface_power",
                surface="top",
                location=dict(starts=(0, 0, 11), ends=(20, 10, 11)),
                params=dict(dim=2, value=1, weight=1),
            )
        ),
    ),
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=-1, weight=1)),
    top=dict(bc=False),
    node=dict(root=True, leaf=True),
    parameterized=dict(variable=False),
)

# Arange the domains in domains_list corresponding to the idx used in domain dict for defining child nodes
# Recommand not to change parameters in global_params except num_params_per_epoch, change it when training DeepOHeat, this is the sample size for your variable
domains_list = [domain_0]
global_params = {
    "loss_fun_type": "norm",
    "num_params_per_epoch": 1,
    "pde_params": dict(type="pde", params=dict(k=0.2, weight=1)),
}

print("Starting training single-case PINN: uniform power defined on the top surface")

# Default device: cuda:0, change it as need, run nvidia-smi to check available GPUs on server
# Using Fourier Features MLP, recommand not to change freq. pi is sufficient for most cases.
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
print("The model used for this case:", model)

dataset = dataio.CuboidGeometryDataIO(domains_list, global_params)
loss_fn = loss_fun.loss_fun_geometry_init(dataset)
val_func = training.val_fn_init(False)

root_path = "./log"
experiment_name = (
    "experiment_1"  # rename it to avoid overwritten if you want to train a new one
)
model_dir = os.path.join(root_path, experiment_name)
lr_decay = True  # if True, multiply 0.9 to lr each epochs_til_decay epochs
epochs_til_decay = 500
epochs_til_val = 500  # visualization each epochs_til_val epochs
epochs = 5000  # total training epochs
lr = 1e-3
epochs_til_checkpoints = 500  # save the model each epochs_til_checkpoints epochs

tic = time.time()
training.train(
    model=model,
    dataset=dataset,
    epochs=epochs,
    lr=lr,
    epochs_til_checkpoints=epochs_til_checkpoints,
    model_dir=model_dir,
    loss_fn=loss_fn,
    val_fn=val_func,
    lr_decay=lr_decay,
    epochs_til_decay=epochs_til_decay,
    epochs_til_val=epochs_til_val,
    device=device,
)
toc = time.time()
print("total training time:", toc - tic)
