import torch
import time
import concurrent.futures
import numpy as np 


def benchmark(model, user, item, n_models=1, n_threads=1, batches_per_thread=1000):
    """
    Record performance statistics for a serialized model and its input example.

    Arguments:
        filename: The serialized torchscript model to load for benchmarking.
        example: An example model input.
        n_models: The number of models to load.
        n_threads: The number of simultaneous threads to execute inferences on.
        batches_per_thread: The number of example batches to run per thread.

    Returns:
        A dictionary of performance statistics.
    """

    # Load models


    # Warmup
    for _ in range(8):
        model(user, item)

    latencies = []

    # Thread task
    def task(model):
        for _ in range(batches_per_thread):
            start = time.time()
            model(user, item)
            finish = time.time()
            latencies.append((finish - start) * 1000)

    # Submit tasks
    begin = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as pool:
        for i in range(n_threads):
            # pool.submit(task, models[i % len(models)])
            pool.submit(task, model)
    end = time.time()

    # Compute metrics
    boundaries = [50, 95, 99]
    percentiles = {}

    for boundary in boundaries:
        name = f'latency_p{boundary}'
        percentiles[name] = np.percentile(latencies, boundary)
    duration = end - begin            
        
    batch_size = 1                
    inferences = len(latencies) * batch_size
    throughput = inferences / duration

    # Metrics
    metrics = {
        'batch_size': batch_size,
        'batches': len(latencies),
        'inferences': inferences,
        'threads': n_threads,
        'models': n_models,
        'duration': duration,
        'throughput': throughput,
        **percentiles,
    }

    display(metrics)

 
def display(metrics):
    """
    Display the metrics produced by `benchmark` function.

    Args:
        metrics: A dictionary of performance statistics.
    """
    pad = max(map(len, metrics)) + 1
    for key, value in metrics.items():

        parts = key.split('_')
        parts = list(map(str.title, parts))
        title = ' '.join(parts) + ":"

        if isinstance(value, float):
            value = f'{value:0.3f}'

        print(f'{title :<{pad}} {value}')
