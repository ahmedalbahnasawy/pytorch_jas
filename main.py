import logging
import argparse
import yaml
import torch
logger = logging.getLogger()
fhandler = logging.FileHandler(filename='mylog.log', mode='a')

def main(args):
  logger.warning('Training ASR model')
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = True 

  with open(args.config, 'r') as f:
      params=yaml.load(f, Loader=yaml.FullLoader)
      print(params)
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str, help='configration file' )
    parser.add_argument('-n','--ngpu',type=int, default=1)
    parser.add_argument('-o','--output-dir', type=str, default=None)
    parser.add_argument('-s','--seed',type=int, default=1234)
    args = parser.parse_args()
    main(args)
else:
    logger.warning('some args are missing')
