# -*- coding: utf-8 -*-

import uuid

from zoology.config import DataConfig, LoggerConfig, ModelConfig, TrainConfig
from zoology.data.associative_recall import MQARConfig

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "monarch_attn" + sweep_id

gate_normalizer = 16
num_layers = 4

VOCAB_SIZE = 8_192

configs = []
for input_seq_len, num_kv_pairs in [
    (512, 64),
]:
    if input_seq_len == 1024:
        batch_size = 64
    elif input_seq_len == 512:
        batch_size = 128
    elif input_seq_len == 256:
        batch_size = 256
    else:
        batch_size = 512

    factory_kwargs = {
        "num_kv_pairs": num_kv_pairs,
        "train_power_a": 0.01,
        "test_power_a": 0.01,
        "random_non_queries": False
    }
    train_configs = [MQARConfig(num_examples=100_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)]
    test_configs = [MQARConfig(num_examples=3_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)]
    data = DataConfig(
        train_configs=train_configs,
        test_configs=test_configs,
        num_train_examples=100_000,
        num_test_examples=3_000,
        vocab_size=VOCAB_SIZE,
        input_seq_len=input_seq_len,
        batch_size=batch_size,
        cache_dir="exp/sabri_data/zg-synthetics",
        builder={
            "name": "zoology.data.associative_recall.multiquery_ar",
            "kwargs": {
                "num_kv_pairs": num_kv_pairs,
                "train_power_a": 0.01,
                "test_power_a": 0.01,
                "random_non_queries": False
            }
        }
    )

    for d_model in [
        64,
        128,
        256,
        512
    ]:
        num_slots = d_model // 4
        for lr in [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]:
            MIXERS = {
                "attention": dict(
                    name="zoology.mixers.attention.MHA",
                    kwargs={
                        "dropout": 0.1,
                        "num_heads": 1
                    },
                ),
                "hyena": dict(
                    name="zoology.mixers.hyena.Hyena",
                    kwargs={
                        "l_max": input_seq_len
                    },
                ),
                "rwkv": dict(
                    name="zoology.mixers.rwkv.RWKVTimeMixer",
                    kwargs={
                        "l_max": input_seq_len,
                    },
                ),
                "base_conv": dict(
                    name="zoology.mixers.base_conv.BaseConv",
                    kwargs={
                        "l_max": input_seq_len,
                        # pass a list of kernel sizes for each of four layers
                        "kernel_size": [3, -1, 3, -1]
                    }
                ),
                "h3": dict(
                    name="zoology.mixers.h3.H3",
                    kwargs={
                        "l_max": input_seq_len,
                        "d_state": input_seq_len,  # makes it mathematically equivalent to Hyena
                        "head_dim": 2
                    }
                ),
                "hybrid": dict(
                    name="zoology.mixers.hybrid.Hybrid",
                    kwargs={
                        "configs": [
                            dict(
                                name="zoology.mixers.base_conv.BaseConv",
                                kwargs={
                                    "l_max": input_seq_len,
                                    # pass a list of kernel sizes for each of four layers
                                    "kernel_size": 3,
                                    "implicit_long_conv": True,
                                }
                            ),
                            dict(
                                name="zoology.mixers.based.Based",
                                kwargs={
                                    "l_max": input_seq_len,
                                    "feature_dim": 8,
                                    "num_key_value_heads": 1,
                                    "num_heads": 1,
                                    "feature_name": "taylor_exp"
                                }
                            )
                        ]
                    }
                ),
                "based": dict(
                    name="zoology.mixers.based.Based",
                    kwargs={
                        "l_max": input_seq_len,
                        "feature_dim": 8,
                        "num_key_value_heads": 1,
                        "num_heads": 1,
                        "feature_name": "taylor_exp"
                    }
                ),
                "mamba": dict(
                    name="zoology.mixers.mamba.Mamba",
                    kwargs={}
                ),
                "gla": dict(
                    name="fla.layers.gla.GatedLinearAttention",
                    kwargs={
                        "mode": "fused_recurrent",
                        "num_heads": 2,
                        "expand_k": 1
                    }
                ),
                "hgrn2": dict(
                    name="fla.layers.hgrn2.HGRN2Attention",
                    kwargs={
                        "mode": "fused_recurrent",
                        "num_heads": 2,
                    }
                ),
                "retnet": dict(
                    name="fla.layers.multiscale_retention.MultiScaleRetention",
                    kwargs={
                        "mode": "fused_recurrent",
                        "num_heads": 2
                    }
                ),
                "gsa": dict(
                    name="fla.layers.gsa.GatedSlotAttention",
                    kwargs={
                        "mode": "fused_recurrent",
                        "num_heads": 4,
                        "num_slots": num_slots,
                        "gate_logit_normalizer": gate_normalizer,
                        "norm_first": False,
                        "scale": None
                    }
                ),
            }

            for sequence_mixer in [
                # "attention",
                # "hyena",
                # "rwkv",
                # "base_conv"
                # "base_conv_explicit",
                # "h3"
                # "base_conv_explicit"
                # "gla",
                # "hgrn2",
                "gsa",
                # "retnet"
                # "mamba"
                # "based",
            ]:
                if 'mamba' in sequence_mixer:
                    block_type = "MambaBlock"
                else:
                    block_type = "TransformerBlock"

                model = ModelConfig(
                    d_model=d_model,
                    n_layers=num_layers,
                    block_type=block_type,
                    max_position_embeddings=input_seq_len if sequence_mixer == "attention" else 0,
                    vocab_size=VOCAB_SIZE,
                    sequence_mixer=MIXERS[sequence_mixer],
                    state_mixer=dict(name="torch.nn.Identity", kwargs={})
                )
                config = TrainConfig(
                    model=model,
                    data=data,
                    learning_rate=lr,
                    max_epochs=64,
                    launch_id=f"{sequence_mixer}-T{input_seq_len}-D{d_model}-L{num_layers}-LR{lr}-KV{num_kv_pairs}",
                    run_id=f"{sequence_mixer}-T{input_seq_len}-D{d_model}-L{num_layers}-LR{lr}-KV{num_kv_pairs}",
                    logger=LoggerConfig(
                        project_name="zoology",
                        entity="yzhangcs"
                    )

                )
                configs.append(config)
