from omegaconf import OmegaConf


def cli():
    print('loading config')
    cli_conf = OmegaConf.from_cli()
    if cli_conf.config_path != None:
        base_conf = OmegaConf.load(cli_conf.config_path)
    else: 
        print('No base config')
        return None
    conf = OmegaConf.merge(base_conf, cli_conf)
    OmegaConf.resolve(conf)
    print(f'config are: {conf}')
    return conf
