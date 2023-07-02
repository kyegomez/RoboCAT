from ray.rllib.algorithms.impala import ImpalaConfig
from ray.tune.logger import pretty_print
import datetime
import os
import tempfile
from ray.tune.logger.unified import UnifiedLogger  # noqa: E402




def custom_log_creator(custom_path, custom_str):

    timestr = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


config = ImpalaConfig()
config = config.training(lr=0.0003, train_batch_size=512)  
config = config.resources(num_gpus=0)  
config = config.rollouts(num_rollout_workers=8)  
config = config.debugging(logger_creator = custom_log_creator(custom_path = 'ray_results', custom_str = 'test'))
config = config.environment(disable_env_checking=True)  
#config = config.environment(env_creator=env_creator)
print(config.to_dict())  
# Build a Algorithm object from the config and run 1 training iteration.
algo = config.build(env='ALE/Kangaroo-v5')  
#algo = config.build()
for i in range(200):
    result = algo.train()
    print(pretty_print(result))