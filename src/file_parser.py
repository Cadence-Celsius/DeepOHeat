import numpy as np
import itertools


def read_power_map(file_path):
    power_map_dict = {}
    with open(file_path) as f:
        power_map = False
        power_map_dim = False
        data = []
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line == "":
                continue

            if not power_map and not power_map_dim:
                line_splitted = line.split(" : ")
                if "Power Unit" in line_splitted:
                    power_map_dict["power_map_dict"] = line_splitted[-1]
                elif "Time Unit" in line_splitted:
                    power_map_dict["Time Unit"] = line_splitted[-1]
                elif "POWER MAP" in line_splitted:
                    power_map_dim = True
            elif power_map_dim:
                power_map_dict["dim"] = tuple([int(i) for i in line.split()])
                power_map_dim = False
                power_map = True
            elif power_map:
                data.append(np.array([float(i) for i in line.split()]).reshape(1, -1))

        power_map_dict["power_map"] = np.concatenate(data, 0)
    f.close()

    return power_map_dict


def res_expand(power_map, scale=1):
    original_shape = power_map.shape
    expanded_shape = tuple([int(i * scale) for i in original_shape])
    new_power_map = np.zeros(expanded_shape)

    for i, j in itertools.product(
        np.arange(original_shape[0]), np.arange(original_shape[1])
    ):
        new_power_map[i * scale : (i + 1) * scale, j * scale : (j + 1) * scale] = (
            power_map[i, j]
        )

    return new_power_map


def convert_interval_to_grid(power_map):
    interval_shape = power_map.shape
    grid_shape = tuple([int(i + 1) for i in interval_shape])

    grid = np.zeros(grid_shape)

    for j in [0, -1]:
        grid[:-1, j] += power_map[:, j]
        grid[1:, j] += power_map[:, j]
        grid[1:-1, j] /= 2

    grid[:-1, 1:-1] += power_map[:, :-1] + power_map[:, 1:]
    grid[1:, 1:-1] += power_map[:, :-1] + power_map[:, 1:]
    grid[1:-1, 1:-1] /= 4
    grid[0, 1:-1] /= 2
    grid[-1, 1:-1] /= 2

    return grid


def from_power_map_to_sensor(file_path, scale=1):
    power_map_dict = read_power_map(file_path)
    power_map = power_map_dict["power_map"]
    power_map_scale = res_expand(power_map, scale)
    sensor = convert_interval_to_grid(power_map_scale)
    sensor /= 0.00625
    return sensor
