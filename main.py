import argparse
from datetime import datetime
from model import *
from data import *
from torch.optim import Adam, lr_scheduler

#Hyperparameters
"""
parser = argparse.ArgumentParser()
print('fine0')

parser.add_argument('--log', default=1, type=int,
                    help='Determine if we log the outputs and experiment configurations to local disk')
parser.add_argument('--path', default=datetime.now().strftime('%Y-%m-%d-%H%M%S'), type=str,
                    help='Default log output path if not specified')
parser.add_argument('--bz', default=32, type=int,
                    help='batch size')
parser.add_argument('--epoch', default=25, type=int,
                    help='number of epochs')
parser.add_argument('--criterion', default='cross_entropy', type=str,
                    help='which loss function to use')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='which optimizer to use')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--dampening', default=0, type=float,
                    help='dampening for momentum')
parser.add_argument('--nesterov', default=False, type=bool,
                    help='enables Nesterov momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay')

parser.add_argument('--lr_scheduler', default='steplr', type=str,
                    help='learning rate scheduler')
parser.add_argument('--step_size', default=7, type=int,
                    help='Period of learning rate decay')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Multiplicative factor of learning rate decay.')
print('fine2')
args = vars(parser.parse_args())
print('fine3')
"""

def main():
    targets, labels = get_data()
    val_data = targets[:len(labels) // 5], labels[:len(labels)// 5]
    train_data = targets[len(labels) // 5 :], labels[len(labels)// 5 :]

    model = baseline()
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr = .001)
    lrs = lr_scheduler.LinearLR(optimizer, total_iters=25)

    model = train_model(model, optimizer, lrs, criterion, train_data, val_data)
    return model
