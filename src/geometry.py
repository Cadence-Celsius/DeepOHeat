import numpy as np
from ordered_set import OrderedSet
import torch
from src.dataio_utils import (
    design_lhs_3d_single_domain,
    find_boundaries_endpoints,
    find_set_by_range_in_subset,
    grid_points_single_domain,
)
from src.geometry_utils import iterate_over_entire_geometry


class Cuboid(object):

    def __init__(self, domain, starting_idx=0, parent=None, parent_boundary=None):

        self.domain, self.domain_step = domain, domain.copy()
        self.name = self.domain["domain_name"]
        self.boundaries_list = ["front", "back", "left", "right", "bottom", "top"]

        if parent is not None and parent_boundary is None:
            raise ValueError(
                f"When creating node with designated parent, one need to provide which boundary this child lies on."
            )

        if parent is not None:
            parent.add_child(parent_boundary, self)
        else:
            self.parent = parent
            self.parent_boundary = parent_boundary

        self.children = {key: [] for key in self.boundaries_list}
        self.starting_idx = starting_idx

        self.tensor, self.conductivity, self.beta, self.q = None, None, None, None
        self.beta_sampling_list = []
        self.whole_set, self.power_set, self.inside_set, self.pde_set = (
            OrderedSet(),
            OrderedSet(),
            OrderedSet(),
            OrderedSet(),
        )
        self.power_points_set_dict = {}
        self.boundaries_set = {key: OrderedSet() for key in self.boundaries_list}

        self.sample(sample_domain=False)
        self.ending_idx = self.starting_idx + self.tensor.shape[0]

    def sample(self, sample_domain=True):

        if sample_domain:
            self.domain_step = self.sample_domain()

        (
            self.tensor,
            self.power_points_set_dict,
            self.inside_set,
            self.boundaries_set,
            self.conductivity,
        ) = design_lhs_3d_single_domain(self.domain_step, self.starting_idx)
        self.power_set = OrderedSet().union(*self.power_points_set_dict.values())

    def sample_grid_points(self, sample_domain=True, sample_mode="random", res=25):
        self.domain_step = (
            self.domain.copy()
            if not sample_domain
            else self.sample_domain(sample_mode=sample_mode)
        )
        self.tensor = grid_points_single_domain(self.domain_step, res)

    def sample_domain(self, sample_mode="random"):
        self.beta_sampling_list = []
        domain_step = self.domain.copy()
        if not self.domain["parameterized"]["variable"]:
            self.beta = np.array([])
            return domain_step

        def sample_param(param_range, param_type):

            if param_type == "continuous":
                if sample_mode == "random":
                    return round(
                        (
                            param_range[0]
                            + np.random.rand(1) * (param_range[1] - param_range[0])
                        ).item(),
                        4,
                    )
                elif sample_mode == "middle":
                    return round((param_range[0] + param_range[1]) / 2, 4)
                elif sample_mode == "low":
                    return param_range[0]
                elif sample_mode == "high":
                    return param_range[1]

            elif param_type == "discrete":
                if sample_mode == "random":
                    return np.random.choice(
                        np.arange(param_range[0], param_range[1] + 1), 1
                    ).item()
                elif sample_mode == "middle":
                    return int(param_range[0] + param_range[1]) / 2
                elif sample_mode == "low":
                    return param_range[0]
                elif sample_mode == "high":
                    return param_range[1]

        for boundary_name, boundary_variable_params in self.domain["parameterized"][
            "param_space"
        ].items():
            if boundary_name == "power":
                for power_id, power_i in boundary_variable_params.items():
                    for variable_param_name, variable_param_dict in power_i.items():
                        param_value = sample_param(
                            variable_param_dict["param_range"],
                            variable_param_dict["type"],
                        )
                        domain_step["power"]["power_map"][power_id]["params"][
                            variable_param_name
                        ] = param_value
                        self.beta_sampling_list.append(param_value)

            else:
                for (
                    variable_param_name,
                    variable_param_dict,
                ) in boundary_variable_params.items():
                    param_value = sample_param(
                        variable_param_dict["param_range"], variable_param_dict["type"]
                    )
                    domain_step[boundary_name]["params"][
                        variable_param_name
                    ] = param_value
                    self.beta_sampling_list.append(param_value)

        self.beta = np.array(self.beta_sampling_list)

        return domain_step

    def find_inside_set(self, whole_set, power_set, boundaries_set):
        return whole_set - OrderedSet().union(*boundaries_set.values()) - power_set

    def update_set(self):
        self.whole_set = OrderedSet(np.arange(self.starting_idx, self.ending_idx))
        self.boundaries_set = self.find_boundries_set(
            self.domain_step, self.tensor, self.boundaries_set
        )
        self.inside_set = self.find_inside_set(
            self.whole_set, self.power_set, self.boundaries_set
        )
        self.pde_set = self.whole_set - self.power_set
        # self.pde_set = self.inside_set

    def find_boundries_set(self, domain, tensor, boundaries_set):

        def find_node_boundary_set(node, adjacent_boundary_name, boundary_set):
            node_boundary_endpoints = find_boundaries_endpoints(
                node.domain_step["geometry"]["starts"],
                node.domain_step["geometry"]["ends"],
            )
            starts, ends = (
                node_boundary_endpoints[adjacent_boundary_name]["starts"],
                node_boundary_endpoints[adjacent_boundary_name]["ends"],
            )
            return find_set_by_range_in_subset(tensor, boundary_set, starts, ends)

        def find_single_boundary_set(boundary_name):
            if not domain[boundary_name]["bc"]:
                return OrderedSet()

            boundary_name_idx = self.boundaries_list.index(boundary_name)
            adjacent_boundary_name = (
                self.boundaries_list[boundary_name_idx - 1]
                if boundary_name_idx % 2
                else self.boundaries_list[boundary_name_idx + 1]
            )

            boundary_set = boundaries_set[boundary_name]
            adj_boundary_set = OrderedSet()

            if not self.is_root() and self.parent_boundary == adjacent_boundary_name:

                adj_boundary_set |= find_node_boundary_set(
                    self.parent, adjacent_boundary_name, boundary_set
                )

            if not self.is_leaf() and self.children[boundary_name] != []:

                for child_node in self.children[boundary_name]:
                    adj_boundary_set |= find_node_boundary_set(
                        child_node, adjacent_boundary_name, boundary_set
                    )

            boundary_set -= adj_boundary_set

            return boundary_set

        return {key: find_single_boundary_set(key) for key in self.boundaries_list}

    def add_child(self, boundary_name, child_node):
        self.children[boundary_name].append(child_node)
        child_node.parent = self
        child_node.parent_boundary = boundary_name

    def to_children(self, boundary_name):
        return self.children[boundary_name]

    def to_child(self, boundary_name, idx):
        return self.children[boundary_name][idx]

    def to_parent(self):
        return self.parent

    def if_last_sibling(self):
        if not self.parent:
            return True, None, None

        self_idx = self.parent.children[self.parent_boundary].index(self)
        if self_idx != len(self.parent.children[self.parent_boundary]) - 1:
            return False, self.parent_boundary, self_idx + 1
        else:
            parent_boundary_idx = self.boundaries_list.index(self.parent_boundary)

            if parent_boundary_idx == len(self.boundaries_list) - 1:
                return True, None, None

            for parent_boundary in self.boundaries_list[parent_boundary_idx + 1 :]:
                if self.parent.children[parent_boundary] != 0:
                    return False, parent_boundary, 0

            return True, None, None

    def to_next_sibling(self, boundary_name, next_idx):
        return self.parent.children[boundary_name][next_idx]

    def to_root(self):
        if self.is_root():
            return self

        prev = self.to_parent()

        while prev.parent is not None:
            prev = prev.to_parent()

        return prev

    def is_root(self):
        return True if self.parent is None else False

    def is_leaf(self):
        for boundary_name in self.boundaries_list:
            if self.children[boundary_name] != []:
                return False

        return True


def create_stacking_cuboidal_geometry(domains_list):
    root = None
    for domain in domains_list:
        if root is None and domain["node"]["root"]:
            root = domain
        elif root is not None and domain["node"]["root"]:
            raise ValueError(f"Only one node can be assigned as the root node")

    if not root:
        raise ValueError(f"Missing the root node, fail to create correct eometry")

    root_node = Cuboid(root)

    if root["node"]["leaf"]:
        return root_node

    all_leaf = False
    current_node = root_node
    child_starting_idx = current_node.ending_idx

    while not all_leaf:

        children_all_leaf = True
        stem_child_boundary_name, stem_child_idx = None, None
        for boundary_name, children_idx_list in current_node.domain["node"][
            "children"
        ].items():
            for child_idx in children_idx_list:
                new_child_node = Cuboid(
                    domains_list[child_idx], starting_idx=child_starting_idx
                )
                current_node.add_child(boundary_name, new_child_node)
                child_starting_idx = new_child_node.ending_idx

                if not domains_list[child_idx]["node"]["leaf"] and children_all_leaf:
                    stem_child_boundary_name, stem_child_idx = (
                        boundary_name,
                        len(current_node.children[boundary_name]) - 1,
                    )

                children_all_leaf &= domains_list[child_idx]["node"]["leaf"]

        if not children_all_leaf:
            current_node = current_node.to_child(
                stem_child_boundary_name, stem_child_idx
            )
        else:
            last_sibling, next_sibling_boundary, next_sibling_idx = (
                current_node.if_last_sibling()
            )

            while last_sibling and current_node.parent is not None:
                current_node = current_node.to_parent()
                last_sibling, next_sibling_boundary, next_sibling_idx = (
                    current_node.if_last_sibling()
                )

            if last_sibling:
                all_leaf = True
            else:
                current_node = current_node.to_next_sibling(
                    next_sibling_boundary, next_sibling_idx
                )

    current_node = current_node.to_root()
    return current_node


def fetch_data(geometry, mode="train", *args):
    tensor_list, conductivity_list, beta_list = [], [], []

    def fetch_single_node(node):
        if mode == "train":
            node.sample(*args)
        elif mode == "eval":
            node.sample_grid_points(*args)

        tensor_list.append(node.tensor)
        beta_list.append(node.beta)
        conductivity_list.append(node.conductivity)

    iterate_over_entire_geometry(geometry, fetch_single_node)
    return (
        torch.tensor(np.concatenate(tensor_list, 0)),
        torch.tensor(np.concatenate(conductivity_list, 0)),
        torch.tensor(np.concatenate(beta_list, 0)),
    )
