import os
import importlib.util

import torch
from isegm.utils.exp import init_experiment
from isegm.utils.parse_args import parse_args_train


def main():
    args = parse_args_train()
    if args.temp_model_path:
        model_script = load_module(args.temp_model_path)
    else:
        model_script = load_module(args.model_path)

    model_base_name = getattr(model_script, 'MODEL_NAME', None)
    
    args.distributed = 'WORLD_SIZE' in os.environ
    cfg = init_experiment(args, model_base_name)

    if cfg.local_rank == 0:
        with open(str(cfg.EXP_PATH/'args.txt'), 'w') as f:
            for key, value in sorted(vars(args).items()):
                f.write(('%s: %s\n' % (str(key), str(value))))
    
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    model_script.main(cfg)


def load_module(script_path):
    spec = importlib.util.spec_from_file_location("model_script", script_path)
    model_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_script)
    
    return model_script


if __name__ == '__main__':
    main()