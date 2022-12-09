from typing import Generator

from trainer.trainer_utils import get_optimizer


class CapacitronOptimizer:
    """Double optimizer class for the Capacitron model."""

    def __init__(self, config: dict, model_params: Generator) -> None:
        self.primary_params, self.secondary_params = self.split_model_parameters(model_params)

        optimizer_names = list(config.optimizer_params.keys())
        optimizer_parameters = list(config.optimizer_params.values())

        self.primary_optimizer = get_optimizer(
            optimizer_names[0],
            optimizer_parameters[0],
            config.lr,
            parameters=self.primary_params,
        )

        self.secondary_optimizer = get_optimizer(
            optimizer_names[1],
            self.extract_optimizer_parameters(optimizer_parameters[1]),
            optimizer_parameters[1]["lr"],
            parameters=self.secondary_params,
        )

        self.param_groups = self.primary_optimizer.param_groups

    def first_step(self):
        self.secondary_optimizer.step()
        self.secondary_optimizer.zero_grad()
        self.primary_optimizer.zero_grad()

    def step(self):
        # Update param groups to display the correct learning rate
        self.param_groups = self.primary_optimizer.param_groups
        self.primary_optimizer.step()

    def zero_grad(self, set_to_none=False):
        self.primary_optimizer.zero_grad(set_to_none)
        self.secondary_optimizer.zero_grad(set_to_none)

    def load_state_dict(self, state_dict):
        self.primary_optimizer.load_state_dict(state_dict[0])
        self.secondary_optimizer.load_state_dict(state_dict[1])

    def state_dict(self):
        return [self.primary_optimizer.state_dict(), self.secondary_optimizer.state_dict()]

    @staticmethod
    def split_model_parameters(model_params: Generator) -> list:
        primary_params = []
        secondary_params = []
        for name, param in model_params:
            if param.requires_grad:
                if name == "capacitron_vae_layer.beta":
                    secondary_params.append(param)
                else:
                    primary_params.append(param)
        return [iter(primary_params), iter(secondary_params)]

    @staticmethod
    def extract_optimizer_parameters(params: dict) -> dict:
        """Extract parameters that are not the learning rate"""
        return {k: v for k, v in params.items() if k != "lr"}
