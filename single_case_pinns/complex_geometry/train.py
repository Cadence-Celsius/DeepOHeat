import time, torch, os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append("../../")
from src import dataio, training, loss_fun, modules

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
    "Starting training single-case PINN: complex geometry and volumetric power defined in the middle layer"
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
print("The model used for this case:", model)

dataset = dataio.CuboidGeometryDataIO(domains_list, global_params)
loss_fn = loss_fun.loss_fun_geometry_init(dataset)
val_func = training.val_fn_init(False)

root_path = "./log"
experiment_name = "experiment_2"
model_dir = os.path.join(root_path, experiment_name)
lr_decay = True
epochs_til_decay = 500
epochs_til_val = 200
epochs = 5000
lr = 1e-3
epochs_til_checkpoints = 500

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
    deeponet=True,
)
toc = time.time()
print("total training time:", toc - tic)
