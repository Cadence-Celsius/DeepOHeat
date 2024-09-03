import time, torch, os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append("../../")
from src import dataio, training, loss_fun, modules

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
    "Starting training DeepOHeat: parameterized top HTC BC with volumetric power defined in the middle layer"
)

for i, domain in enumerate(domains_list):
    print("domain %d:" % i, domain)

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

dataset = dataio.CuboidGeometryDataIO(domains_list, global_params)
loss_fn = loss_fun.loss_fun_geometry_init(dataset)
val_func = training.val_fn_init(False)

root_path = "./log"
experiment_name = "experiment_1"
model_dir = os.path.join(root_path, experiment_name)
lr_decay = True
epochs_til_decay = 500
epochs_til_val = 50
epochs = 5000
lr = 1e-3
epochs_til_checkpoints = 200

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
