from ....equivariant.nn import GroupTensor

import numpy as np
import torch
from torch.nn import Module

from abc import ABC, abstractmethod

from typing import List, Tuple, Any

__all__ = ["EquivariantModule"]


class EquivariantModule(Module, ABC):
    def __init__(self):
        r"""
        Abstract base class for all equivariant modules.

        An :class:`~EquivariantModule` is a subclass of :class:`torch.nn.Module`.
        It follows that any subclass of :class:`~EquivariantModule` needs to implement the
        :meth:`~nn.EquivariantModule.forward` method.
        With respect to a general :class:`torch.nn.Module`, an *equivariant module* implements a *typed* function as
        both its input and its output are associated with specific :class:`~nn.FieldType` s.
        Therefore, usually, the inputs and the outputs of an *equivariant module* are not just instances of
        :class:`torch.Tensor` but :class:`~nn.GroupTensor` s.

        As a subclass of :class:`torch.nn.Module`, it supports most of the commonly used methods (e.g.
        :meth:`torch.nn.Module.to`, :meth:`torch.nn.Module.cuda`, :meth:`torch.nn.Module.train` or
        :meth:`torch.nn.Module.eval`)

        Many equivariant modules implement a :meth:`~nn.EquivariantModule.export` method which converts the module
        to *eval* mode and returns a pure PyTorch implementation of it.
        This can be used after training to efficiently deploy the model without, for instance, the overhead of the
        automatic type checking performed by all the modules in this library.

        .. warning ::

            Not all modules implement this feature yet.
            If the :meth:`~nn.EquivariantModule.export` method is called in a module which does not implement it
            yet, a :class:`NotImplementedError` is raised.
            Check the documentation of each individual module to understand if the method is implemented.

        Attributes:
            ~.in_type (FieldType): type of the :class:`~nn.GroupTensor` expected as input
            ~.out_type (FieldType): type of the :class:`~nn.GroupTensor` returned as output

        """
        super(EquivariantModule, self).__init__()

        # FieldType: type of the :class:`~nn.GroupTensor` expected as input
        self.in_type = None

        # FieldType: type of the :class:`~nn.GroupTensor` returned as output
        self.out_type = None

    @abstractmethod
    def forward(self, *input):
        pass

    @abstractmethod
    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        r"""
        Compute the shape the output tensor which would be generated by this module when a tensor with shape
        ``input_shape`` is provided as input.

        Args:
            input_shape (tuple): shape of the input tensor

        Returns:
            shape of the output tensor

        """
        pass

    def check_equivariance(
        self, x: torch.Tensor = None, atol: float = 1e-4, rtol: float = 1e-4
    ) -> List[Tuple[Any, float]]:
        r"""

        Method that automatically tests the equivariance of the current module.
        The default implementation of this method relies on :meth:`nn.GroupTensor.transform` and uses the
        the group elements in :attr:`~nn.FieldType.testing_elements`.

        This method can be overwritten for custom tests.

        Returns:
            a list containing containing for each testing element a pair with that element and the corresponding
            equivariance error

        """
        if x is None:
            c = self.in_type.size
            x = torch.randn(32, c, 29, 29).cuda()
            x = GroupTensor(x, self.in_type)

        errors = []
        for el in self.out_type.testing_elements:
            out1 = self(x).transform(el).tensor.detach().cpu().numpy()
            out2 = self(x.transform(el)).tensor.detach().cpu().numpy()

            errs = out1 - out2
            errs = np.abs(errs).reshape(-1)
            print(
                f"Group {el}: err max: {errs.max()} - err mean: {errs.mean()} - err var: {errs.var()}"
            )

            assert np.allclose(
                out1, out2, atol=atol, rtol=rtol
            ), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(
                el, errs.max(), errs.mean(), errs.var()
            )

            errors.append((el, errs.mean()))

        return errors

    def export(self):
        r"""
        Export recursively each submodule to a normal PyTorch module and set to "eval" mode.

        .. warning ::

            Not all modules implement this feature yet.
            If the :meth:`~nn.EquivariantModule.export` method is called in a module which does not implement it
            yet, a :class:`NotImplementedError` is raised.
            Check the documentation of each individual module to understand if the method is implemented.

        .. warning ::
            Since most modules do not use the `coords` attribute of the input :class:`~nn.GroupTensor`,
            once converted, they will only expect `tensor` but not `coords` in input.
            There is no standard behavior for modules that explicitly use `coords`, so check their specific
            documentation.

        """

        raise NotImplementedError(
            "Conversion of equivariant module {} into PyTorch module is not supported yet".format(
                self.__class__
            )
        )
