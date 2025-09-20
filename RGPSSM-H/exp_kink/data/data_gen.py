
from .synthetic import syn_data_generation
from .utils import save_pickle, load_pickle, set_seed

set_seed(0)

def load_data(filename):
    return load_pickle(filename)


def generate_data(filename):
    num_experiments = 5
    var_list = [0.008, 0.08, 0.8]

    data = {}
    for var in var_list:
        for i in range(num_experiments):
            d = data_single(var, i)
            key = f'var = {var:.3f}, i = {i}'
            data[key] = d

            print(key)

    save_pickle(data, filename)


def data_single(var_observation, i):
    func = 'kinkfunc'
    number_ips = 15  # number of inducing points
    episode = 30
    seq_len = 20
    process_noise_sd = 0.05
    ips, state_np, observe_np = syn_data_generation(func=func, traj_len=episode*seq_len, process_noise_sd=process_noise_sd,
                                                        observation_noise_sd=var_observation**0.5, number_ips=number_ips)
    data = {'x': state_np, 'y': observe_np, 'ips': ips}

    return data


