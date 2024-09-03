import time, torch, os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.append("../../")
from src import dataio_deeponet, training_deeponet, loss_fun_deeponet, modules

# In this example the domain is not as functioning as before.
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
    # Define this power is only to do eval during training. This power is not used in training.
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
    # Define the BCs as ususal. Make sure the top surface has bc=True, no type and params is needed for top.
    front=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    back=dict(bc=True, type="adiabatics", params=dict(dim=1, weight=1)),
    left=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    right=dict(bc=True, type="adiabatics", params=dict(dim=0, weight=1)),
    bottom=dict(bc=True, type="htc", params=dict(dim=2, k=0.2, direction=-1, weight=1)),
    top=dict(bc=True),
    node=dict(root=True, leaf=True),
    # Here parameterized dict is not quite functioning, I suggest keep it unchanged.
    parameterized=dict(variable=False),
)

# Larger num_params_per_epoch is better. Here 50 is close to OOM on a single GPU.
domains_list = [domain_0]
global_params = {
    "loss_fun_type": "norm",
    "num_params_per_epoch": 50,
    "pde_params": dict(type="pde", params=dict(k=0.2, weight=1)),
}

print("Starting training DeepOHeat: arbitrary 2D power map")

for i, domain in enumerate(domains_list):
    print("domain %d:" % i, domain)

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

# The two variables control the functional space of the power map
# larger var indicates wider range of power value
# larger len_scale indicates smoother power distribution
var = 1
len_scale = 0.3

print("var: {}, len scale: {}".format(var, len_scale))

# Here mesh coordinates are used. LHS design also works.
dataset = dataio_deeponet.DeepONetMeshDataIO(
    domains_list, global_params, dim=2, var=var, len_scale=len_scale
)
loss_fn = loss_fun_deeponet.mesh_loss_fun_geometry_init(dataset)
val_func = training_deeponet.val_fn_init(False)

root_path = "./log"
experiment_name = "experiment_1"
model_dir = os.path.join(root_path, experiment_name)
lr_decay = True
epochs_til_decay = 500
epochs_til_val = 50
epochs = 10000
lr = 1e-3
epochs_til_checkpoints = 1000

tic = time.time()
training_deeponet.train_mesh(
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
