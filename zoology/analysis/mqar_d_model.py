# -*- coding: utf-8 -*-

import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from zoology.analysis.utils import fetch_wandb_runs


def plot(
    df: pd.DataFrame,
    max_seq_len: int = 512,
):
    df.dropna(axis=0, inplace=True, subset=['model.sequence_mixer.name'])
    plt.ticklabel_format(style='plain', axis='x')
    print(
        df.sort_values([
            "model.sequence_mixer.name",
            "model.state_mixer.name",
            'model.sequence_mixer.kwargs.gate_logit_normalizer',
            'model.n_layers',
            "model.d_model",
            "learning_rate",
        ], ascending=[True, True, True, True, True, True]).groupby([
            "model.sequence_mixer.name",
            'model.state_mixer.name',
            # 'model.sequence_mixer.kwargs.gate_logit_normalizer',
            'model.n_layers',
            "model.d_model",
            "learning_rate",
            'data.input_seq_len',
        ])["valid/accuracy"].max().reset_index()
    )

    plot_df = df.groupby([
        "model.sequence_mixer.name",
        "model.d_model",
        'data.input_seq_len',
    ])["valid/accuracy"].max().reset_index()
    print(plot_df)
    plot_df["model.d_model"] = plot_df["model.d_model"].apply(lambda x: str(int(x)))
    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=plot_df[plot_df["data.input_seq_len"] <= max_seq_len],
        y="valid/accuracy",
        col="data.input_seq_len",
        x="model.d_model",
        hue="model.sequence_mixer.name",
        kind="line",
        marker="o",
        # height=2.25,
        aspect=1,
    )
    g.set(ylabel="Accuracy", xlabel="Model dimension")
    # Set custom x-ticks
    # ticks = [64, 128, 256, 512]  # Modify this list as needed
    # for ax in g.axes.flat:
    #     # This will keep the tick labels as integers rather than in scientific notation
    #     # ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    #     # ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())  # <---- Added
    #     ax.set_xticks(ticks)
    #     # ax.set_xticklabels(ticks)

    # Set custom y-ticks
    y_ticks = [0, 0.25, 0.5, 0.75, 1.0]
    for ax in g.axes.flat:
        ax.set_yticks(y_ticks)

    for ax, title in zip(g.axes.flat, g.col_names):
        ax.set_title(f"Sequence Length: {int(title)}")


if __name__ == "__main__":
    df = fetch_wandb_runs(
        run_id=[
            'gsa-T512-D64-L4-LR0.0001-KV64',
            'gsa-T512-D64-L4-LR0.0005-KV64',
            'gsa-T512-D64-L4-LR0.001-KV64',
            'gsa-T512-D64-L4-LR0.002-KV64',
            'gsa-T512-D64-L4-LR0.005-KV64',
            'gsa-T512-D64-L4-LR0.01-KV64',
            'gsa-T512-D128-L4-LR0.0001-KV64',
            'gsa-T512-D128-L4-LR0.0005-KV64',
            'gsa-T512-D128-L4-LR0.0005-KV64',
            'gsa-T512-D128-L4-LR0.001-KV64',
            'gsa-T512-D128-L4-LR0.002-KV64',
            'gsa-T512-D128-L4-LR0.002-KV64',
            'gsa-T512-D128-L4-LR0.005-KV64',
            'gsa-T512-D128-L4-LR0.005-KV64',
            'gsa-T512-D128-L4-LR0.01-KV64',
            'gsa-T512-D128-L4-LR0.01-KV64',
            'gsa-T512-D256-L4-LR0.0001-KV64',
            'gsa-T512-D256-L4-LR0.0005-KV64',
            'gsa-T512-D256-L4-LR0.001-KV64',
            'gsa-T512-D256-L4-LR0.002-KV64',
            'gsa-T512-D256-L4-LR0.005-KV64',
            'gsa-T512-D256-L4-LR0.01-KV64',
            'gsa-T512-D512-L4-LR0.0001-KV64',
            'gsa-T512-D512-L4-LR0.0001-KV64',
            'gsa-T512-D512-L4-LR0.0005-KV64',
            'gsa-T512-D512-L4-LR0.0005-KV64',
            'gsa-T512-D512-L4-LR0.001-KV64',
            'gsa-T512-D512-L4-LR0.001-KV64',
            'gsa-T512-D512-L4-LR0.002-KV64',

            'hgrn2-T512-D64-L4-LR0.0001-KV64',
            'hgrn2-T512-D64-L4-LR0.0005-KV64',
            'hgrn2-T512-D64-L4-LR0.001-KV64',
            'hgrn2-T512-D64-L4-LR0.002-KV64',
            'hgrn2-T512-D64-L4-LR0.005-KV64',
            'hgrn2-T512-D64-L4-LR0.01-KV64',
            'hgrn2-T512-D128-L4-LR0.0001-KV64',
            'hgrn2-T512-D128-L4-LR0.0005-KV64',
            'hgrn2-T512-D128-L4-LR0.001-KV64',
            'hgrn2-T512-D128-L4-LR0.002-KV64',
            'hgrn2-T512-D128-L4-LR0.005-KV64',
            'hgrn2-T512-D128-L4-LR0.01-KV64',
            'hgrn2-T512-D256-L4-LR0.0001-KV64',
            'hgrn2-T512-D256-L4-LR0.0005-KV64',
            'hgrn2-T512-D256-L4-LR0.001-KV64',
            'hgrn2-T512-D256-L4-LR0.002-KV64',
            'hgrn2-T512-D256-L4-LR0.005-KV64',
            'hgrn2-T512-D256-L4-LR0.01-KV64',
            'hgrn2-T512-D512-L4-LR0.0001-KV64',
            'hgrn2-T512-D512-L4-LR0.0005-KV64',
            'hgrn2-T512-D512-L4-LR0.001-KV64',
            'hgrn2-T512-D512-L4-LR0.002-KV64',
            'hgrn2-T512-D512-L4-LR0.005-KV64',
            'hgrn2-T512-D512-L4-LR0.01-KV64',

            'mamba-T512-D128-L4-LR0.0001-KV64',
            'mamba-T512-D128-L4-LR0.0005-KV64',
            'mamba-T512-D64-L4-LR0.0001-KV64',
            'mamba-T512-D64-L4-LR0.0005-KV64',
            'mamba-T512-D64-L4-LR0.001-KV64',
            'mamba-T512-D64-L4-LR0.002-KV64',
            'mamba-T512-D64-L4-LR0.005-KV64',
            'mamba-T512-D64-L4-LR0.01-KV64',
            'mamba-T512-D128-L4-LR0.001-KV64',
            'mamba-T512-D128-L4-LR0.002-KV64',
            'mamba-T512-D128-L4-LR0.005-KV64',
            'mamba-T512-D128-L4-LR0.01-KV64',
            'mamba-T512-D256-L4-LR0.0001-KV64',
            'mamba-T512-D256-L4-LR0.0005-KV64',
            'mamba-T512-D256-L4-LR0.001-KV64',
            'mamba-T512-D256-L4-LR0.002-KV64',
            'mamba-T512-D256-L4-LR0.005-KV64',
            'mamba-T512-D256-L4-LR0.01-KV64',
            'mamba-T512-D512-L4-LR0.0001-KV64',
            'mamba-T512-D512-L4-LR0.0005-KV64',
            'mamba-T512-D512-L4-LR0.001-KV64',
            'mamba-T512-D512-L4-LR0.002-KV64',
            'mamba-T512-D512-L4-LR0.005-KV64',
            'mamba-T512-D512-L4-LR0.01-KV64',
        ],
        project_name="zoology",
    )

    plot(df=df, max_seq_len=512)
    plt.savefig(sys.argv[1])
