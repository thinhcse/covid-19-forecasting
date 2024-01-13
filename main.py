import yaml, os
from models.model import model
from utils.data_utils import data_preparation
from utils.model_utils import train, test
from argparse import ArgumentParser, BooleanOptionalAction
from utils.visualization import infectious_plot, recovered_plot, deceased_plot

parser = ArgumentParser()
parser.add_argument("--train", dest="is_train", help="To turn on training mode", action=BooleanOptionalAction)
parser.add_argument("--config-file", dest="config_path", help="Path to configuration file", metavar="str",
                    default=os.path.join(os.getcwd(), "configs", "config.yaml"))

args = parser.parse_args()
is_train = args.is_train
config_path = args.config_path

with open(config_path) as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)

predictor = model().set_configs(configs)
data = data_preparation(configs)

data_train_ld, data_val_ld, data_test_ld = data[0]
start_idx_train, start_idx_val, start_idx_test = data[1]
time_stamps_train, time_stamps_val, time_stamps_test = data[2]

if __name__ == "__main__":

    if is_train:
        train(predictor, data_train_ld, data_val_ld, configs)
    else:
        predictor.load_parameters(os.path.join(os.getcwd(), configs["pretrain"]))

    preds, truths, times, error = test(predictor, data_test_ld, configs, time_stamps_test)
    print("MAE: %.3f" %(error * 1E6))

    infectious_plot(preds, truths, times, configs)
    recovered_plot(preds, truths, times, configs)
    deceased_plot(preds, truths, times, configs)