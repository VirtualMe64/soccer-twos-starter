import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='output.png')
parser.add_argument('--x-label', type=str, default='Total Timesteps')
parser.add_argument('--x-stat', type=str, default='timesteps_total')
parser.add_argument('--y-label', type=str, default='Average Episode Length')
parser.add_argument('--y-stat', type=str, default='episode_len_mean')
parser.add_argument('experiments', nargs=argparse.REMAINDER)

args = parser.parse_args()

def parse_experiment(experiment):
    out = []
    for root, _, files in os.walk(experiment):
        for file in files:
            if file != 'progress.csv':
                continue
            
            path = os.path.join(root, file)
            df = pd.read_csv(path)
            out.append(df)
    df = pd.concat(out, ignore_index=True)
    df = df.sort_values(args.x_stat)
    return df

dfs = []
for experiment_dir in args.experiments[0::2]:
    dfs.append(parse_experiment(experiment_dir))

plt.figure()
plt.style.use('ggplot')
plt.xlabel(args.x_label)
plt.ylabel(args.y_label)
for df, experiment_name in zip(dfs, args.experiments[1::2]):
    plt.plot(df[args.x_stat], df[args.y_stat], label=experiment_name)

plt.legend()
plt.savefig(args.output)