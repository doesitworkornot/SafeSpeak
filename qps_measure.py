import gc
import torch
import sys
import time
import numpy as np
from scipy.stats import sem, t

from model_description import Model
from dataset_preparing import get_data_for_evaldataset, EvalDataset, get_dataloaders, pad


@torch.inference_mode
def run_inference(data_loader, model, device):
    model.eval()
    fname_list = []
    score_list = []

    for batch_x, utt_id in progressbar(data_loader, prefix='skimming through set'):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_out = model.forward(batch_x)
            prob = torch.nn.functional.softmax(batch_out, dim=1)
            batch_score = (prob[:, 1]).data.cpu().numpy().ravel()
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
    assert len(fname_list) == len(score_list)


def progressbar(it, prefix="", size=60, out=sys.stdout):
    count = len(it)
    start = time.time()

    def show(j):
        x = int(size * j / count)
        remaining = ((time.time() - start) / j) * (count - j)
        passing = time.time() - start
        mins_pas, sec_pass = divmod(passing, 60)
        time_pas = f"{int(mins_pas):02}:{sec_pass:05.2f}"

        mins, sec = divmod(remaining, 60)
        time_str = f"{int(mins):02}:{sec:05.2f}"

        print(f"{prefix}[{u'█' * x}{('.' * (size - x))}] {j}/{count} time {time_pas} / {time_str}", end='\r', file=out,
              flush=True)

    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_qps_with_ci(device, paths, cfg, num_runs=10, confidence=0.95):
    model = Model(device, paths[1])  # path to wav2vec
    model.load_state_dict(torch.load(paths[0], map_location=device))  # path to model after w2v
    print(f'\nNumber of parameters is {count_parameters(model)}\n')

    path_wav = './dataset'
    eval_ids = get_data_for_evaldataset(path_wav)

    eval_dataset = EvalDataset(eval_ids, path_wav, pad)
    dataloader = get_dataloaders(eval_dataset, cfg)

    print(f'Starting to measure time on {device}\n\n')

    if cfg['ci']:
        qps_values = []
        for run in range(num_runs):
            print(f'Run {run + 1}/{num_runs}...')
            start = time.time()
            run_inference(dataloader, model, device)
            end = time.time()
            qps = len(eval_dataset) / (end - start)
            qps_values.append(qps)
            print(f'QPS for run {run + 1}: {qps:.4f}')
        
        mean_qps = np.mean(qps_values)
        stderr = sem(qps_values)
        t_value = t.ppf((1 + confidence) / 2, num_runs - 1)
        ci_range = t_value * stderr

        print(f'\nQPS (mean): {mean_qps:.4f}')
        print(f'95% Confidence Interval: [{mean_qps - ci_range:.4f}, {mean_qps + ci_range:.4f}]')
        print(f'with deviation equals {ci_range:.2f}')
    else:
        start = time.time()
        run_inference(dataloader, model, device)
        end = time.time()
        qps = len(eval_dataset) / (end - start)
        print(f'QPS for {device} is {qps:.4f}')

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()



if __name__ == '__main__':
    cfg = {
        "batch_size": 8,
        "num_class": 2,
        "num_workers": 8,
        "ci": True
    }
    paths = ['./weights/w2v2_scoof.pth', './weights/xlsr2_300m.pt']

    device = 'cuda'
    calculate_qps_with_ci(device, paths, cfg)
    
    
    device = 'cpu'
    calculate_qps_with_ci(device, paths, cfg)