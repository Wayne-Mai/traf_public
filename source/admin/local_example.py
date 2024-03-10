
import os
from datetime import datetime

class EnvironmentSettings:
    def __init__(self, data_root='', debug=False, arg_log_dir='logs'):
        # Current date and time
        current_date = datetime.now().strftime("%m-%d-%Y")
        current_time = datetime.now().strftime("%H-%M-%p")

        # Base directory for logs
        base_log_dir = arg_log_dir

        # Create a directory for today's date
        daily_log_dir = os.path.join(base_log_dir, current_date)

        # Create a specific directory for this experiment based on the current time
        experiment_log_dir = os.path.join(daily_log_dir, current_time)

        # Set the directory paths
        self.log_dir = experiment_log_dir
        self.workspace_dir = os.path.join(experiment_log_dir, 'workspace')    # For saving network checkpoints
        self.tensorboard_dir = os.path.join(experiment_log_dir, 'tensorboard')    # For tensorboard files
        self.pretrained_networks = self.workspace_dir    # For saving pre-trained networks
        self.eval_dir = os.path.join(experiment_log_dir, 'eval')    # For saving evaluations

        # Data directories
        if data_root=='':
            self.llff = 'data/nerf_llff_data'
            self.dtu = 'data/rs_dtu_4/DTU'
            self.dtu_depth = 'data/'
            self.dtu_mask = 'data/submission_data/idrmasks'
            self.replica = 'data/Replica'
            self.precomputed = 'data/precomputed'
        else:
            self.llff = f'{data_root}/nerf_llff_data'
            self.dtu = f'{data_root}/rs_dtu_4/DTU'
            self.dtu_depth = f'{data_root}/' # the dataset loader will append /Depth to it
            self.dtu_mask = f'{data_root}/submission_data/idrmasks'
            self.replica = f'{data_root}/Replica'
            self.precomputed = f'{data_root}/precomputed'

        