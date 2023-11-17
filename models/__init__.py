from models.regada import ReGaDaModel
from models.utils import print_model, print_model_size


def get_model(
    model_args,
    dset,
):
    model_name = model_args["name"]

    if model_name == "regada":
        model = ReGaDaModel(dset=dset, **model_args)
    else:
        raise NotImplementedError("Model does not exist!")

    print_model(model)
    print_model_size(model)

    return model
