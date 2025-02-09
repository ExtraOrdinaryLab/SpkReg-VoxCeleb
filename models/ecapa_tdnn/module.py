import torch
import torch.nn as nn


class NeuralModule(nn.Module):

    @property
    def num_weights(self):
        """
        Utility property that returns the total number of parameters of NeuralModule.
        """
        return self._num_weights()

    @torch.jit.ignore
    def _num_weights(self):
        num: int = 0
        for p in self.parameters():
            if p.requires_grad:
                num += p.numel()
        return num

    def freeze(self) -> None:
        r"""
        Freeze all params for inference.

        This method sets `requires_grad` to False for all parameters of the module.
        It also stores the original `requires_grad` state of each parameter in a dictionary,
        so that `unfreeze()` can restore the original state if `partial=True` is set in `unfreeze()`.
        """
        grad_map = {}

        for pname, param in self.named_parameters():
            # Store the original grad state
            grad_map[pname] = param.requires_grad
            # Freeze the parameter
            param.requires_grad = False

        # Store the frozen grad map
        if not hasattr(self, '_frozen_grad_map'):
            self._frozen_grad_map = grad_map
        else:
            self._frozen_grad_map.update(grad_map)

        self.eval()

    def unfreeze(self, partial: bool = False) -> None:
        """
        Unfreeze all parameters for training.

        Allows for either total unfreeze or partial unfreeze (if the module was explicitly frozen previously with `freeze()`).
        The `partial` argument is used to determine whether to unfreeze all parameters or only the parameters that were
        previously unfrozen prior `freeze()`.

        Example:
            Consider a model that has an encoder and a decoder module. Assume we want the encoder to be frozen always.

            ```python
            model.encoder.freeze()  # Freezes all parameters in the encoder explicitly
            ```

            During inference, all parameters of the model should be frozen - we do this by calling the model's freeze method.
            This step records that the encoder module parameters were already frozen, and so if partial unfreeze is called,
            we should keep the encoder parameters frozen.

            ```python
            model.freeze()  # Freezes all parameters in the model; encoder remains frozen
            ```

            Now, during fine-tuning, we want to unfreeze the decoder but keep the encoder frozen. We can do this by calling
            `unfreeze(partial=True)`.

            ```python
            model.unfreeze(partial=True)  # Unfreezes only the decoder; encoder remains frozen
            ```

        Args:
            partial: If True, only unfreeze parameters that were previously frozen. If the parameter was already frozen
                when calling `freeze()`, it will remain frozen after calling `unfreeze(partial=True)`.
        """
        if partial and not hasattr(self, '_frozen_grad_map'):
            raise ValueError("Cannot unfreeze partially without first freezing the module with `freeze()`")

        for pname, param in self.named_parameters():
            if not partial:
                # Unfreeze all parameters
                param.requires_grad = True
            else:
                # Unfreeze only parameters that were previously frozen

                # Check if the parameter was frozen
                if pname in self._frozen_grad_map:
                    param.requires_grad = self._frozen_grad_map[pname]
                else:
                    # Log a warning if the parameter was not found in the frozen grad map
                    print(
                        f"Parameter {pname} not found in list of previously frozen parameters. "
                        f"Unfreezing this parameter."
                    )
                    param.requires_grad = True

        # Clean up the frozen grad map
        if hasattr(self, '_frozen_grad_map'):
            delattr(self, '_frozen_grad_map')

        self.train()