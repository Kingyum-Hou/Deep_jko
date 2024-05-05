from . import resnet


# model config
model_dict = {
    'ResNet': resnet,
}


def get_model(model_name, args, model_dict=model_dict):
    return model_dict[model_name].Model(args)
