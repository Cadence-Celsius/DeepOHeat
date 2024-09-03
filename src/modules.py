import torch
import torch.nn as nn
from collections import OrderedDict


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


def xavier_init(layer):
    with torch.no_grad():
        if type(layer) == nn.Linear:
            if hasattr(layer, "weight"):
                nn.init.xavier_normal_(layer.weight)
        else:
            raise TypeError(f"Expecting nn.Linear got type={type(layer)} instead")


class FCBlock(nn.Module):

    def __init__(
        self,
        out_features=1,
        in_features=3,
        hidden_features=20,
        num_hidden_layers=3,
        nonlinearity="sine",
        device=None,
    ):
        super().__init__()

        nl_init_dict = dict(
            sine=(Sine(), xavier_init),
            silu=(nn.SiLU(), xavier_init),
            tanh=(nn.Tanh(), xavier_init),
            relu=(nn.ReLU(), xavier_init),
        )
        nl, init = nl_init_dict[nonlinearity]

        self.net = OrderedDict()

        for i in range(num_hidden_layers + 2):
            if i == 0:
                self.net["fc1"] = nn.Linear(
                    in_features=in_features, out_features=hidden_features
                )
                self.net["nl1"] = nl
            elif i != num_hidden_layers + 1:
                self.net["fc%d" % (i + 1)] = nn.Linear(
                    in_features=hidden_features, out_features=hidden_features
                )
                self.net["nl%d" % (i + 1)] = nl

            else:
                self.net["fc%d" % (i + 1)] = nn.Linear(
                    in_features=hidden_features, out_features=out_features
                )

            init(self.net["fc%d" % (i + 1)])

        self.net = nn.Sequential(self.net)

        if device:
            self.net.to(device)

    def forward(self, x):
        return self.net(x)


class DNN(nn.Module):

    def __init__(
        self,
        out_features=1,
        in_features=3,
        hidden_features=20,
        num_hidden_layers=3,
        nonlinearity="sine",
        device=None,
    ):
        super().__init__()

        self.net = FCBlock(
            out_features=out_features,
            in_features=in_features,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        coords = coords_org

        output = self.net(coords)
        return {"model_in": coords_org, "model_out": output}


class FFN(nn.Module):

    def __init__(
        self,
        out_features=1,
        in_features=3,
        hidden_features=20,
        num_hidden_layers=3,
        nonlinearity="sine",
        device=None,
        freq=torch.pi,
        std=1,
        freq_trainable=True,
    ):
        super().__init__()

        self.net = FCBlock(
            out_features=out_features,
            in_features=hidden_features,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )

        self.freq = nn.Parameter(
            torch.ones(1, device=device) * freq, requires_grad=freq_trainable
        )
        self.fourier_features = nn.Parameter(
            torch.zeros(in_features, int(hidden_features / 2), device=device).normal_(
                0, std
            )
            * self.freq,
            requires_grad=False,
        )

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        coords = coords_org

        ff_input = torch.concat(
            [
                torch.sin(torch.matmul(coords, self.fourier_features)),
                torch.cos(torch.matmul(coords, self.fourier_features)),
            ],
            -1,
        )

        output = self.net(ff_input)
        return {"model_in": coords_org, "model_out": output}


class ModifiedFC(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        nonlinearity="sine",
        transform=True,
        activate=True,
    ):
        super().__init__()

        nl_init_dict = dict(
            sine=(Sine(), xavier_init),
            silu=(nn.SiLU(), xavier_init),
            tanh=(nn.Tanh(), xavier_init),
            relu=(nn.ReLU(), xavier_init),
        )

        self.nl, self.init_fun = nl_init_dict[nonlinearity]
        self.transform = transform
        self.activate = activate

        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.init_fun(self.fc)

    def forward(self, x, trans_1, trans_2):

        if self.transform:
            output = self.nl(self.fc(x))
            return (1 - output) * trans_1 + output * trans_2

        elif self.activate:
            return self.nl(self.fc(x))

        else:
            return self.fc(x)


class ModifiedFCBlock(nn.Module):

    def __init__(
        self,
        out_features=1,
        in_features=3,
        hidden_features=20,
        num_hidden_layers=3,
        nonlinearity="sine",
        device=None,
    ):
        super().__init__()

        nl_init_dict = dict(
            sine=(Sine(), xavier_init),
            silu=(nn.SiLU(), xavier_init),
            tanh=(nn.Tanh(), xavier_init),
            relu=(nn.ReLU(), xavier_init),
        )
        nl, init = nl_init_dict[nonlinearity]

        self.net = []

        for i in range(num_hidden_layers + 2):
            if i == 0:
                self.net.append(
                    ModifiedFC(
                        in_features=in_features,
                        out_features=hidden_features,
                        nonlinearity=nonlinearity,
                        transform=False,
                        activate=True,
                    )
                )
            elif i != num_hidden_layers + 1:
                self.net.append(
                    ModifiedFC(
                        in_features=hidden_features,
                        out_features=hidden_features,
                        nonlinearity=nonlinearity,
                        transform=True,
                    )
                )
            else:
                self.net.append(
                    ModifiedFC(
                        in_features=hidden_features,
                        out_features=out_features,
                        transform=False,
                        activate=False,
                    )
                )

        self.transform_layer_1 = nn.Linear(
            in_features=in_features, out_features=hidden_features
        )
        init(self.transform_layer_1)
        self.transform_layer_2 = nn.Linear(
            in_features=in_features, out_features=hidden_features
        )
        init(self.transform_layer_2)

        self.net = nn.ModuleList(self.net)

        if device:
            self.net.to(device)
            self.transform_layer_1.to(device)
            self.transform_layer_2.to(device)

    def forward(self, x):
        trans_1 = self.transform_layer_1(x)
        trans_2 = self.transform_layer_2(x)

        for net_i in self.net:
            x = net_i(x, trans_1, trans_2)
        return x


class ModifiedFCBlockFourier(nn.Module):

    def __init__(
        self,
        out_features=1,
        in_features=3,
        hidden_features=20,
        num_hidden_layers=3,
        nonlinearity="sine",
        device=None,
    ):
        super().__init__()

        nl_init_dict = dict(
            sine=(Sine(), xavier_init),
            silu=(nn.SiLU(), xavier_init),
            tanh=(nn.Tanh(), xavier_init),
            relu=(nn.ReLU(), xavier_init),
        )
        nl, init = nl_init_dict[nonlinearity]

        self.net = []

        for i in range(num_hidden_layers + 2):
            if i == 0:
                self.net.append(
                    ModifiedFC(
                        in_features=in_features,
                        out_features=hidden_features,
                        nonlinearity=nonlinearity,
                        transform=False,
                        activate=True,
                    )
                )
            elif i != num_hidden_layers + 1:
                self.net.append(
                    ModifiedFC(
                        in_features=hidden_features,
                        out_features=hidden_features,
                        nonlinearity=nonlinearity,
                        transform=True,
                    )
                )
            else:
                self.net.append(
                    ModifiedFC(
                        in_features=hidden_features,
                        out_features=out_features,
                        transform=False,
                        activate=False,
                    )
                )

        self.transform_layer_1 = nn.Linear(
            in_features=hidden_features, out_features=hidden_features
        )
        init(self.transform_layer_1)
        self.transform_layer_2 = nn.Linear(
            in_features=hidden_features, out_features=hidden_features
        )
        init(self.transform_layer_2)

        self.net = nn.ModuleList(self.net)

        if device:
            self.net.to(device)
            self.transform_layer_1.to(device)
            self.transform_layer_2.to(device)

    def forward(self, x, fourier_features):
        trans_1 = self.transform_layer_1(fourier_features)
        trans_2 = self.transform_layer_2(fourier_features)

        for net_i in self.net:
            x = net_i(x, trans_1, trans_2)
        return x


class ModifiedDNN(nn.Module):

    def __init__(
        self,
        out_features=1,
        in_features=3,
        hidden_features=20,
        num_hidden_layers=3,
        nonlinearity="sine",
        device=None,
    ):
        super().__init__()

        self.net = ModifiedFCBlock(
            out_features=out_features,
            in_features=in_features,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        coords = coords_org

        output = self.net(coords)
        return {"model_in": coords_org, "model_out": output}


class ModifiedFFN(nn.Module):

    def __init__(
        self,
        out_features=1,
        in_features=3,
        hidden_features=20,
        num_hidden_layers=3,
        nonlinearity="sine",
        device=None,
        freq=torch.pi,
        std=1,
        freq_trainable=True,
    ):
        super().__init__()

        self.net = ModifiedFCBlockFourier(
            out_features=out_features,
            in_features=in_features,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )

        self.freq = nn.Parameter(
            torch.ones(1, device=device) * torch.pi, requires_grad=freq_trainable
        )
        self.fourier_features = nn.Parameter(
            torch.zeros(in_features, int(hidden_features / 2), device=device).normal_(
                0, std
            )
            * self.freq,
            requires_grad=False,
        )

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        coords = coords_org

        ff_input = torch.concat(
            [
                torch.sin(torch.matmul(coords, self.fourier_features)),
                torch.cos(torch.matmul(coords, self.fourier_features)),
            ],
            -1,
        )

        output = self.net(coords, ff_input)
        return {"model_in": coords_org, "model_out": output}


class DeepONet(nn.Module):

    def __init__(
        self,
        trunk_in_features=3,
        trunk_hidden_features=128,
        branch_in_features=1,
        branch_hidden_features=20,
        inner_prod_features=50,
        num_branch_hidden_layers=3,
        num_trunk_hidden_layers=3,
        nonlinearity="silu",
        freq=torch.pi,
        std=1,
        freq_trainable=True,
        device=None,
    ):
        super().__init__()

        self.branch = FCBlock(
            out_features=inner_prod_features,
            in_features=branch_in_features,
            hidden_features=branch_hidden_features,
            num_hidden_layers=num_branch_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )

        self.trunk = FCBlock(
            out_features=inner_prod_features,
            in_features=trunk_hidden_features,
            hidden_features=trunk_hidden_features,
            num_hidden_layers=num_trunk_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )

        self.freq = nn.Parameter(
            torch.ones(1, device=device) * freq, requires_grad=freq_trainable
        )
        self.fourier_features = nn.Parameter(
            torch.zeros(
                trunk_in_features, int(trunk_hidden_features / 2), device=device
            ).normal_(0, std)
            * self.freq,
            requires_grad=False,
        )
        self.b_0 = nn.Parameter(
            torch.zeros(1, device=device).uniform_(), requires_grad=True
        )

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        coords = coords_org

        beta = model_input["beta"].clone().detach().requires_grad_(True)

        ff_input = torch.concat(
            [
                torch.sin(torch.matmul(coords, self.fourier_features)),
                torch.cos(torch.matmul(coords, self.fourier_features)),
            ],
            -1,
        )

        output_1 = self.trunk(ff_input)
        output_2 = self.branch(beta)

        output = torch.sum(output_1 * output_2, 1).reshape(-1, 1) + self.b_0
        return {"model_in": coords_org, "model_out": output}


class BranchNetList(nn.Module):

    def __init__(self, net_arc, num_nets, *args):
        super().__init__()

        self.net_list = nn.ModuleList([net_arc(*args) for i in range(num_nets)])

    def forward(self, x, trunk_output):
        output = trunk_output
        for i, branch_i in enumerate(self.net_list):
            output *= branch_i(x[:, i].reshape(-1, 1))

        return output


class MIONet(nn.Module):

    def __init__(
        self,
        trunk_in_features=3,
        trunk_hidden_features=128,
        branch_in_features=1,
        branch_hidden_features=20,
        inner_prod_features=50,
        num_hidden_layers=3,
        nonlinearity="silu",
        freq=torch.pi,
        std=1,
        freq_trainable=True,
        device=None,
    ):
        super().__init__()

        self.branch = BranchNetList(
            FCBlock,
            branch_in_features,
            inner_prod_features,
            1,
            branch_hidden_features,
            num_hidden_layers,
            nonlinearity,
            device,
        )

        self.trunk = FCBlock(
            out_features=inner_prod_features,
            in_features=trunk_hidden_features,
            hidden_features=trunk_hidden_features,
            num_hidden_layers=num_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )

        self.freq = nn.Parameter(
            torch.ones(1, device=device) * freq, requires_grad=freq_trainable
        )
        self.fourier_features = nn.Parameter(
            torch.zeros(
                trunk_in_features, int(trunk_hidden_features / 2), device=device
            ).normal_(0, std)
            * self.freq,
            requires_grad=False,
        )
        self.b_0 = nn.Parameter(
            torch.zeros(1, device=device).uniform_(), requires_grad=True
        )

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        coords = coords_org

        beta = model_input["beta"].clone().detach().requires_grad_(True)

        ff_input = torch.concat(
            [
                torch.sin(torch.matmul(coords, self.fourier_features)),
                torch.cos(torch.matmul(coords, self.fourier_features)),
            ],
            -1,
        )

        output = (
            torch.sum(self.branch(beta, self.trunk(ff_input)), 1).reshape(-1, 1)
            + self.b_0
        )

        return {"model_in": coords_org, "model_out": output}


class FFONet(nn.Module):

    def __init__(
        self,
        trunk_in_features=3,
        trunk_hidden_features=128,
        branch_in_features=1,
        branch_hidden_features=20,
        inner_prod_features=50,
        num_branch_hidden_layers=3,
        num_trunk_hidden_layers=3,
        nonlinearity="silu",
        trunk_freq=torch.pi,
        branch_freq=torch.pi,
        std=1,
        freq_trainable=True,
        device=None,
    ):
        super().__init__()

        self.branch = FCBlock(
            out_features=inner_prod_features,
            in_features=branch_hidden_features,
            hidden_features=branch_hidden_features,
            num_hidden_layers=num_branch_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )

        self.trunk = FCBlock(
            out_features=inner_prod_features,
            in_features=trunk_hidden_features,
            hidden_features=trunk_hidden_features,
            num_hidden_layers=num_trunk_hidden_layers,
            nonlinearity=nonlinearity,
            device=device,
        )

        self.trunk_freq = nn.Parameter(
            torch.ones(1, device=device) * trunk_freq, requires_grad=freq_trainable
        )
        self.branch_freq = nn.Parameter(
            torch.ones(1, device=device) * branch_freq, requires_grad=freq_trainable
        )

        self.branch_fourier_features = nn.Parameter(
            torch.zeros(
                branch_in_features, int(branch_hidden_features / 2), device=device
            ).normal_(0, std)
            * self.branch_freq,
            requires_grad=False,
        )
        self.trunk_fourier_features = nn.Parameter(
            torch.zeros(
                trunk_in_features, int(trunk_hidden_features / 2), device=device
            ).normal_(0, std)
            * self.trunk_freq,
            requires_grad=False,
        )
        self.b_0 = nn.Parameter(
            torch.zeros(1, device=device).uniform_(), requires_grad=True
        )

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        coords = coords_org

        beta = model_input["beta"].clone().detach().requires_grad_(True)

        trunk_ff_input = torch.concat(
            [
                torch.sin(torch.matmul(coords, self.trunk_fourier_features)),
                torch.cos(torch.matmul(coords, self.trunk_fourier_features)),
            ],
            -1,
        )
        branch_ff_input = torch.concat(
            [
                torch.sin(torch.matmul(beta, self.branch_fourier_features)),
                torch.cos(torch.matmul(beta, self.branch_fourier_features)),
            ],
            -1,
        )

        output_1 = self.trunk(trunk_ff_input)
        output_2 = self.branch(branch_ff_input)

        output = torch.sum(output_1 * output_2, 1).reshape(-1, 1) + self.b_0
        return {"model_in": coords_org, "model_out": output}
