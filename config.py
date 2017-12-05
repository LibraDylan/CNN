class Config(object):
  pass

def get_config(is_train):
  config = Config()
  if is_train:
    config.batch_size = 6
    config.im_size = [256, 512]
    config.lr = 1e-4
    config.iteration = 1000

    config.tmp_dir = "tmp"
    config.ckpt_dir = "ckpt"
  else:
    config.batch_size = 6
    config.im_size = [256, 512]

    config.result_dir = "result"
    config.ckpt_dir = "ckpt"
  return config
