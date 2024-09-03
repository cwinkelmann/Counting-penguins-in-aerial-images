import logging

from model.vgg import vgg19
import argparse
import os
from dataset.dataset import Crowd
from torch.utils.data import DataLoader
import torch
import numpy as np


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='processed_data/test',
                        help='The path to data directory')
    parser.add_argument('--saved-model', default='history/penguin-15_2024_08_15/best_model.pth',
                        help='model directory')
    parser.add_argument('--device', default='0',
                        help='assign device')
    parser.add_argument('--pretrained', default=False,
                        help='the path to the pretrained model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_arg()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()

    datasets = Crowd(args.data_dir, 448, 8, 'val') # TODO why is this 448 when the training is 256?
    dataloader = DataLoader(datasets, 1, shuffle=False, pin_memory=False)

    model = vgg19()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        logging.warning('GPU is not available')
        device = torch.device('cpu')
        device_count = 0

    model.to(device)

    model.load_state_dict(torch.load(args.saved_model, device))

    out_file = './validation_result.csv'
    out = open(out_file, 'w')

    out.write('name,true,pred,diff\n')
    print('-' * 25 + 'Model loaded!' + '-' * 25)
    model.eval()
    overall = []
    true_total = []
    preds_total = []

    for inputs, points, name in dataloader:
        inputs = inputs.to(device)
        
        assert inputs.size(0) == 1
        with torch.set_grad_enabled(False):
            den, bg = model(inputs)
            #  TODO: den is the density map, not the count
            # TODO: what is bg?
            true = points.item()

            # TODO plot the density map
            pred = (((den * (bg >= 0.5)))).sum().item()
            diff = true - pred

            overall.append(abs(diff))
            true_total.append(true)
            preds_total.append(pred)

            print(name[0], true, pred, diff)
            out.write('{},{},{},{}\n'.format(name[0], true, pred, diff))
            print('-' * 60)
    out.close()

    print(f"Overall absolute Difference: {np.array(overall).sum()}")
    print(f"Overall True Counts: {np.array(true_total).sum()}")
    print(f"Overall Preds Counts: {np.array(preds_total).sum()}")

    print(f"Average Overall absolute Difference: {np.array(overall).mean()}")
    print(f"Average Overall True Counts: {np.array(true_total).mean()}")
    print(f"Average Overall Preds Counts: {np.array(preds_total).mean()}")





