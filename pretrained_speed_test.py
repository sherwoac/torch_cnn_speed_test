import functools
import time
import torch
import torchvision.models
import multiprocessing
import tqdm

# exclude_models = ['densenet121', 'densenet161', 'densenet169', 'densenet201']
exclude_models = ['mnasnet1_3', 'mnasnet0_75']
number_of_cycles = 100


def print_gpu_info(_):
    print('# GPU Info')
    print(f'|torch.cuda.is_available():| {torch.cuda.is_available()}|')
    print(f'|torch.cuda.current_device():| {torch.cuda.current_device()}|')
    print(f'|device number|device name|')
    print(f'|---|---|')
    for device_number in range(torch.cuda.device_count()):
        print(f'|{device_number}|{torch.cuda.get_device_name(device_number)}|')


def get_gpu_device():
    return torch.cuda.current_device()


def create_test_batch(batch_size):
    return torch.rand(size=(batch_size, 3, 224, 224))


def invoke_model(model_name, dev):
    model = eval(f'torchvision.models.{model_name}(pretrained=True)')
    model.to(dev)
    return model


def run_model_test(model, dev, batch_size, cycles=number_of_cycles):
    test_data = create_test_batch(batch_size=batch_size).to(dev)
    _ = model(test_data)
    start_time = time.time()
    try:
        for i in range(cycles):
            _ = model(test_data)
        elapsed = time.time() - start_time
        return elapsed
    except RuntimeError as re:
        print(re)

    return 0.


def run_model(model_name, dev, batch_size):
    try:
        model = invoke_model(model_name, dev)
        return run_model_test(model, dev, batch_size)
    except ValueError as ve:
        pass
    except NotImplementedError as ne:
        pass  # eg. pre trained not supported
    except RuntimeError as re:
        pass  # eg. Calculated padded input size per channel: ..

    return 0.


def run_model_new_device(model_input):
    dev = get_gpu_device()
    model_name, batch_size = model_input
    return run_model(model_name, dev, batch_size)


if __name__ == "__main__":
    batch_sizes = list(map(functools.partial(pow, 2), range(10)))
    pool = multiprocessing.Pool(processes=1)
    pool.map(print_gpu_info, [()])
    elapsed_times_model_batch_size = {}
    models_to_run = []
    for batch_size in batch_sizes:
        for model_name in dir(torchvision.models):
            beginning_letter = model_name[0]
            if beginning_letter != '_' and not beginning_letter.isupper() and model_name not in exclude_models:
                model_class = eval(f'torchvision.models.{model_name}')
                if callable(model_class):
                    models_to_run.append((model_name, batch_size))

    elapsed_times_per_model = tqdm.tqdm(pool.map(run_model_new_device, models_to_run))
    elapsed_times_model_batch_size.update(dict(zip(models_to_run, elapsed_times_per_model)))

    print('|batch size|model name|(s)|(fps)|')
    print('|---|---|---|---|')
    for (model_name, batch_size), model_elapsed_time in elapsed_times_model_batch_size.items():
        if model_elapsed_time:
            print(f'|{batch_size}|{model_name}|{model_elapsed_time:.1f}|{int((batch_size * number_of_cycles)/model_elapsed_time):,}|')
        else:
            print(f'|{batch_size}|{model_name}|-|-|')


