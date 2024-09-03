import numpy as np
from ordered_set import OrderedSet
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
from gstools import SRF, Gaussian


def sample_field(xyz_axes, srf, mesh_type="unstructured"):
    field = srf(xyz_axes, mesh_type=mesh_type)
    return field.reshape(-1)


def sample_grf_model(dim=3, var=1, len_scale=0.3, pool=10000000, seed=None):
    model = Gaussian(dim=dim, var=var, len_scale=len_scale)
    if seed is None:
        seed = np.random.randint(pool)
    srf = SRF(model, seed=seed)

    return seed, srf


def fixed_mesh_grid_3d(starts, ends, num_intervals):
    xyz_axes = [
        np.linspace(starts[i], ends[i], num_intervals[i] + 1)
        for i in range(len(num_intervals))
    ]
    mesh = np.meshgrid(*xyz_axes, indexing="ij")
    coords = np.concatenate([item.reshape(-1, 1) for item in mesh], 1)

    return coords


def lhs_sampling_3d(starts, ends, num_points):
    end_points = np.array([[starts[i], ends[i]] for i in range(3)])
    sampling = LHS(xlimits=end_points)
    coords = sampling(num_points)
    return coords


def grid_points_single_domain(domain, res=25):
    geometry = domain["geometry"]
    starts = geometry["starts"]
    ends = geometry["ends"]

    num_points = [50, 50, 50]
    return fixed_mesh_grid_3d(starts, ends, num_points)


def find_boundaries_endpoints(starts, ends):
    boundaries_dict = dict(
        front=dict(starts=starts, ends=[ends[0], starts[1], ends[2]]),
        back=dict(starts=[starts[0], ends[1], starts[2]], ends=ends),
        left=dict(starts=starts, ends=[starts[0], ends[1], ends[2]]),
        right=dict(starts=[ends[0], starts[1], starts[2]], ends=ends),
        bottom=dict(starts=starts, ends=[ends[0], ends[1], starts[2]]),
        top=dict(starts=[starts[0], starts[1], ends[2]], ends=ends),
    )

    return boundaries_dict


def find_set_by_range(tensor, starting_idx, starts, ends, dim=3):
    if dim == 3:
        return OrderedSet(
            np.where(
                (tensor[..., 0] >= starts[0])
                & (tensor[..., 0] <= ends[0])
                & (tensor[..., 1] >= starts[1])
                & (tensor[..., 1] <= ends[1])
                & (tensor[..., 2] >= starts[2])
                & (tensor[..., 2] <= ends[2])
            )[0]
            + starting_idx
        )
    else:
        return OrderedSet(
            np.where(
                (tensor[..., 0] >= starts[0])
                & (tensor[..., 0] <= ends[0])
                & (tensor[..., 1] >= starts[1])
                & (tensor[..., 1] <= ends[1])
            )[0]
            + starting_idx
        )


def find_set_by_range_in_subset(tensor, tensor_set, starts, ends, dim=3):
    points = tensor[tensor_set, :]
    if dim == 3:
        idx = np.where(
            (points[..., 0] >= starts[0])
            & (points[..., 0] <= ends[0])
            & (points[..., 1] >= starts[1])
            & (points[..., 1] <= ends[1])
            & (points[..., 2] >= starts[2])
            & (points[..., 2] <= ends[2])
        )[0]

        return OrderedSet(tensor_set[idx])
    else:
        idx = np.where(
            (points[..., 0] >= starts[0])
            & (points[..., 0] <= ends[0])
            & (points[..., 1] >= starts[1])
            & (points[..., 1] <= ends[1])
        )[0]

        return OrderedSet(tensor_set[idx])


def sample_training_data_single_domain(domain, srf, dim=3, starting_idx=0):
    geometry = domain["geometry"]
    starts = geometry["starts"]
    ends = geometry["ends"]
    num_intervals = geometry["num_intervals"]
    num_pde_points = geometry["num_pde_points"]
    num_single_bc_points = geometry["num_single_bc_points"]
    xyz_axes = [np.linspace(starts[i], ends[i], num_intervals[i] + 1) for i in range(3)]

    sensors = sample_field(xyz_axes[:dim], srf, "structured")

    conductivity_dist = domain["conductivity_dist"]
    uneven_conductivity = conductivity_dist["uneven_conductivity"]
    background_conductivity = conductivity_dist["background_conductivity"]

    sampling_parts = ["front", "back", "left", "right", "bottom", "top"]
    boundary_points_set_dict = {key: OrderedSet() for key in sampling_parts}
    inside_points_set = OrderedSet()
    cumu_idx = starting_idx

    tensor = lhs_sampling_3d(starts, ends, num_pde_points)
    cumu_idx_temp = int(cumu_idx + num_pde_points)
    inside_points_set |= OrderedSet(np.arange(cumu_idx, cumu_idx_temp))
    cumu_idx = cumu_idx_temp

    boundaries_endpoints_dict = find_boundaries_endpoints(starts, ends)
    for boundary_name, boundary_endpoints in boundaries_endpoints_dict.items():
        boundary_starts, boundary_ends = (
            boundary_endpoints["starts"],
            boundary_endpoints["ends"],
        )
        if boundary_name == "top" and dim == 2:
            tensor = np.concatenate(
                [
                    tensor,
                    lhs_sampling_3d(
                        boundary_starts, boundary_ends, int(num_single_bc_points * 10)
                    ),
                ],
                0,
            )
            cumu_idx_temp = int(cumu_idx + num_single_bc_points * 10)
        else:
            tensor = np.concatenate(
                [
                    tensor,
                    lhs_sampling_3d(
                        boundary_starts, boundary_ends, num_single_bc_points
                    ),
                ],
                0,
            )
            cumu_idx_temp = int(cumu_idx + num_single_bc_points)

        boundary_points_set_dict[boundary_name] |= OrderedSet(
            np.arange(cumu_idx, cumu_idx_temp)
        )
        cumu_idx = cumu_idx_temp

    if dim == 3:
        power_map = sample_field([tensor[:, i].reshape(-1) for i in range(3)], srf)
    elif dim == 2:
        power_map = np.zeros(tensor.shape[0])
        power_face_idx_set = boundary_points_set_dict["top"]
        power_face_points = tensor[power_face_idx_set, :]
        power_map[power_face_idx_set] = sample_field(
            [power_face_points[:, i].reshape(-1) for i in range(2)], srf
        )

    conductivity = np.ones(tensor.shape[0]) * background_conductivity
    whole_set = OrderedSet(np.arange(tensor.shape[0]))
    if uneven_conductivity:
        materials = conductivity_dist["materials"]
        materials_set_dict = {}

        for material_id, material in materials.items():
            material_starts_idx, material_ends_idx = material["location"].values()
            material_starts = [xyz_axes[i][material_starts_idx[i]] for i in range(3)]
            material_ends = [xyz_axes[i][material_ends_idx[i]] for i in range(3)]
            material_points_set = find_set_by_range_in_subset(
                tensor, whole_set, material_starts, material_ends
            )
            materials_set_dict[material_id] = material_points_set

        material_points_set_last = OrderedSet()
        for material_id, material_points_set in materials_set_dict.items():
            material_points_set -= material_points_set_last
            conductivity[material_points_set] = materials[material_id]["value"]
            material_points_set_last |= material_points_set

    return (
        sensors,
        tensor,
        power_map,
        inside_points_set,
        boundary_points_set_dict,
        conductivity,
    )


def sample_eval_data_single_domain(domain, dim=3):
    geometry = domain["geometry"]
    starts = geometry["starts"]
    ends = geometry["ends"]
    num_intervals = geometry["num_intervals"]
    xyz_axes = [np.linspace(starts[i], ends[i], num_intervals[i] + 1) for i in range(3)]

    sensor_mesh = fixed_mesh_grid_3d(starts[:dim], ends[:dim], num_intervals[:dim])
    sensors = np.zeros(sensor_mesh.shape[0])
    whole_set = OrderedSet(np.arange(sensor_mesh.shape[0]))

    power = domain["power"]

    if power["bc"]:

        for power_id, power_i in power["power_map"].items():
            power_i_location = power_i["location"]

            power_i_starts_idx, power_i_ends_idx = power_i_location.values()
            power_i_starts = [xyz_axes[i][power_i_starts_idx[i]] for i in range(3)]
            power_i_ends = [xyz_axes[i][power_i_ends_idx[i]] for i in range(3)]
            power_i_set = find_set_by_range_in_subset(
                sensor_mesh, whole_set, power_i_starts, power_i_ends, dim=dim
            )

            if power_i_set == OrderedSet():
                raise ValueError(f"No sensor location is available for {power_id}")

            sensors[power_i_set] = power_i["params"]["value"]

    eval_coords = grid_points_single_domain(domain)

    return sensors, eval_coords


def sample_sensor_as_coords_train_data_single_domain(domain, srf, dim=3):
    geometry = domain["geometry"]
    starts = geometry["starts"]
    ends = geometry["ends"]
    num_intervals = geometry["num_intervals"]
    xyz_axes = [np.linspace(starts[i], ends[i], num_intervals[i] + 1) for i in range(3)]
    sampling_parts = ["front", "back", "left", "right", "bottom", "top"]
    boundary_points_set_dict = {key: OrderedSet() for key in sampling_parts}

    sensors = sample_field(xyz_axes[:dim], srf, "structured")

    tensor = fixed_mesh_grid_3d(starts, ends, num_intervals)
    whole_set = OrderedSet(np.arange(tensor.shape[0]))

    conductivity_dist = domain["conductivity_dist"]
    background_conductivity = conductivity_dist["background_conductivity"]

    boundaries_endpoints_dict = find_boundaries_endpoints(starts, ends)
    for boundary_name, boundary_endpoints in boundaries_endpoints_dict.items():
        boundary_starts, boundary_ends = (
            boundary_endpoints["starts"],
            boundary_endpoints["ends"],
        )
        boundary_points_set_dict[boundary_name] |= find_set_by_range_in_subset(
            tensor, whole_set, boundary_starts, boundary_ends
        )

    for boundary_name in sampling_parts[:4]:
        boundary_points_set_dict[boundary_name] -= boundary_points_set_dict["top"]
        boundary_points_set_dict[boundary_name] -= boundary_points_set_dict["bottom"]

    conductivity = np.ones(tensor.shape[0]) * background_conductivity

    return sensors, tensor, boundary_points_set_dict, conductivity


def design_lhs_3d_single_domain(domain, starting_idx=0):

    geometry = domain["geometry"]
    starts = geometry["starts"]
    ends = geometry["ends"]
    num_intervals = geometry["num_intervals"]
    num_pde_points = geometry["num_pde_points"]
    num_single_bc_points = geometry["num_single_bc_points"]
    xyz_axes = [np.linspace(starts[i], ends[i], num_intervals[i] + 1) for i in range(3)]

    power = domain["power"]

    conductivity_dist = domain["conductivity_dist"]
    uneven_conductivity = conductivity_dist["uneven_conductivity"]
    background_conductivity = conductivity_dist["background_conductivity"]

    sampling_parts = ["front", "back", "left", "right", "bottom", "top"]
    boundary_points_set_dict = {key: OrderedSet() for key in sampling_parts}
    power_points_set_dict = {}
    inside_points_set = OrderedSet()
    cumu_idx = starting_idx

    tensor = lhs_sampling_3d(starts, ends, num_pde_points)
    cumu_idx_temp = int(cumu_idx + num_pde_points)
    inside_points_set |= OrderedSet(np.arange(cumu_idx, cumu_idx_temp))
    cumu_idx = cumu_idx_temp

    boundaries_endpoints_dict = find_boundaries_endpoints(starts, ends)
    for boundary_name, boundary_endpoints in boundaries_endpoints_dict.items():
        boundary_starts, boundary_ends = (
            boundary_endpoints["starts"],
            boundary_endpoints["ends"],
        )
        boundary_points = lhs_sampling_3d(
            boundary_starts, boundary_ends, num_single_bc_points
        )
        tensor = np.concatenate([tensor, boundary_points], 0)

        cumu_idx_temp = int(cumu_idx + num_single_bc_points)
        boundary_points_set_dict[boundary_name] |= OrderedSet(
            np.arange(cumu_idx, cumu_idx_temp)
        )
        cumu_idx = cumu_idx_temp

    if power["bc"]:
        num_power_points_per_volume = power["num_power_points_per_volume"]
        num_power_points_per_surface = power["num_power_points_per_surface"]
        num_power_points_per_cell = power["num_power_points_per_cell"]
        area_surface = num_intervals[0] * num_intervals[1]

        def log_increase(total_num, per_num, total_capacity, capacity):
            alpha = (total_num - per_num) / np.log(total_capacity + 1)
            if capacity >= total_capacity:
                return int(total_num * capacity / total_capacity)
            else:
                return int(alpha * np.log(capacity + 1) + per_num)

        def define_num_power_points(capacity):
            if power_i_type == "volumetric_power":
                return int(capacity * num_power_points_per_volume)
            elif power_i_type == "surface_power":
                return log_increase(
                    num_power_points_per_surface,
                    num_power_points_per_cell,
                    area_surface,
                    capacity,
                )

        inside_in_power_set = OrderedSet()
        boundary_in_power_set_dict = {key: OrderedSet() for key in sampling_parts}

        for power_id, power_i in power["power_map"].items():
            power_i_location = power_i["location"]
            power_i_type = power_i["type"]
            in_power_i_set = OrderedSet()

            if power_i_type != "volumetric_power" and power_i_type != "surface_power":
                raise TypeError(
                    f'Do not support {power_i_type}, please choose from {"surface_power"} and {"volumetric_power"}'
                )

            power_i_starts_idx, power_i_ends_idx = (
                power_i_location["starts"],
                power_i_location["ends"],
            )
            power_i_capacity = np.prod(
                [
                    (
                        power_i_ends_idx[i] - power_i_starts_idx[i]
                        if power_i_ends_idx[i] - power_i_starts_idx[i] > 0
                        else 1
                    )
                    for i in range(3)
                ]
            )
            num_power_i_points = define_num_power_points(power_i_capacity)

            power_i_starts = [xyz_axes[i][power_i_starts_idx[i]] for i in range(3)]
            power_i_ends = [xyz_axes[i][power_i_ends_idx[i]] for i in range(3)]
            power_i_points = lhs_sampling_3d(
                power_i_starts, power_i_ends, num_power_i_points
            )

            if power_i_type == "volumetric_power":
                in_power_i_set |= find_set_by_range_in_subset(
                    tensor, inside_points_set, power_i_starts, power_i_ends
                )
                inside_in_power_set |= in_power_i_set

            elif power_i_type == "surface_power":
                power_i_surface = power_i["surface"]
                in_power_i_set |= find_set_by_range_in_subset(
                    tensor,
                    boundary_points_set_dict[power_i_surface],
                    power_i_starts,
                    power_i_ends,
                )
                boundary_in_power_set_dict[power_i_surface] |= in_power_i_set

            cumu_idx_temp = int(cumu_idx + num_power_i_points)
            power_i_set = OrderedSet(np.arange(cumu_idx, cumu_idx_temp))
            cumu_idx = cumu_idx_temp

            tensor = np.concatenate([tensor, power_i_points], 0)
            power_i_set |= in_power_i_set
            power_points_set_dict[power_id] = power_i_set

        inside_points_set -= inside_in_power_set
        boundary_points_set_dict = {
            key: boundary_points_set_dict[key] - boundary_in_power_set_dict[key]
            for key in sampling_parts
        }

    conductivity = np.ones(tensor.shape[0]) * background_conductivity
    whole_set = OrderedSet(np.arange(tensor.shape[0]))
    if uneven_conductivity:
        materials = conductivity_dist["materials"]
        materials_set_dict = {}

        for material_id, material in materials.items():
            material_starts_idx, material_ends_idx = material["location"].values()
            material_starts = [xyz_axes[i][material_starts_idx[i]] for i in range(3)]
            material_ends = [xyz_axes[i][material_ends_idx[i]] for i in range(3)]
            material_points_set = find_set_by_range_in_subset(
                tensor, whole_set, material_starts, material_ends
            )
            materials_set_dict[material_id] = material_points_set

        material_points_set_last = OrderedSet()
        for material_id, material_points_set in materials_set_dict.items():
            material_points_set -= material_points_set_last
            conductivity[material_points_set] = materials[material_id]["value"]
            material_points_set_last |= material_points_set

    return (
        tensor,
        power_points_set_dict,
        inside_points_set,
        boundary_points_set_dict,
        conductivity,
    )
