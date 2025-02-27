import torch
import gpytorch
from PMTO.models import CustomGPModel

# x = torch.rand(30, 5)
# y = torch.rand(30, 2)
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
#
# my_model = CustomGPModel(x, y, likelihood, 3, 2)
# print(list(my_model.named_parameters()))
# print(my_model.covar_module.base_kernel.get_lengthscales())

contexts = torch.load("data/context_8_2.pth")
print(contexts)