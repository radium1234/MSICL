from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed, set_color
from parser import parse_args

from micro import micro
from trainer import Trainer
from lattice import Lattice
from lightgcn import LightGCN
from msicl import MSICL
import os
import torch
import numpy as np





def run_single_model(args):
    # configurations initialization
    config = Config(
        model=MSICL,
        dataset=args.dataset, 
        config_file_list=args.config_file_list
    )
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)
    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    # model loading and initialization
    if args.model=="micro":
        model = micro(config, train_data.dataset).to(config['device'])
    elif args.model == "lightgcn":
        model = LightGCN(config, train_data.dataset).to(config['device'])
    elif args.model == "lattice":
        model = Lattice(config, train_data.dataset).to(config['device'])
    elif args.model == "msicl":
        if args.cluster !=0 and not os.path.exists('dataset/'+args.dataset+'/lightrecord.npy') :
            model = LightGCN(config, train_data.dataset).to(config['device'])
            logger.info(model)

            trainer = Trainer(config, model)
            best_valid_score, best_valid_result = trainer.fit(
                train_data, valid_data, saved=True, show_progress=config['show_progress']
            )

            # model evaluation
            test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
            logger.info(set_color('test result', 'yellow') + f': {test_result}')
            np.save('dataset/'+args.dataset + "/lightrecord.npy", model.restore_item_e.detach().cpu().numpy())
        model = MSICL(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization



    trainer = Trainer(config, model)
    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')


if __name__ == '__main__':


    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    torch.cuda.set_device(args.gpu_id)
    # Config files
    args.config_file_list = [
        'properties/overall.yaml',
    ]
    args.config_file_list.append('properties/'+args.model+'.yaml')

    assert args.dataset in ['amazon-sports-outdoors','amazon-clothing-shoes-jewelry',"amazon-toys-games"]

    args.config_file_list.append(f'properties/{args.dataset}.yaml')

    run_single_model(args)
    print(args.dataset)
    print(args.model)

