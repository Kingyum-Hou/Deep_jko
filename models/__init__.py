from . import resnet, phi


# model config
model_dict = {
    'ResNet': resnet,
    'Phi': phi,
}


def get_model(model_name, args, model_dict=model_dict):
    return model_dict[model_name].Model(args)
