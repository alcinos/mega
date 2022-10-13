#!/usr/bin/env python

import sweep
from sweep import hyperparam


# Normal usage:
# python -u train.py ${DATA} \
#     --seed 0 --ddp-backend c10d --find-unused-parameters \
#     -a mega_lra_listop --task lra-text --input-type text \
#     --encoder-layers 6 --n-dim 16 --chunk-size $CHUNK \
#     --activation-fn 'silu' --attention-activation-fn 'softmax' \
#     --norm-type 'layernorm' --sen-rep-type 'mp' \
#     --criterion lra_cross_entropy --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
#     --optimizer adam --lr 0.001 --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
#     --dropout 0.1 --attention-dropout 0.0 --act-dropout 0.0 --weight-decay 0.01 \
#     --batch-size 64 --sentence-avg --update-freq 1 --max-update 90000 --max-sentences-valid 256 \
#     --lr-scheduler linear_decay --total-num-update 90000 --end-learning-rate 0.0 \
#     --warmup-updates 3000 --warmup-init-lr '1e-07' --keep-last-epochs 1 --required-batch-size-multiple 1 \
#     --save-dir ${SAVE} --log-format simple --log-interval 100 --num-workers 0

def get_grid(args):
    """
    Replicates the `16-bit+cumul+2x lr` results from Table 1 of
    "Scaling Neural Machine Translation" (https://arxiv.org/abs/1806.00187)
    """
    return [
        hyperparam("--seed", 0),
        hyperparam("--ddp-backend", "c10d"),
        hyperparam("--find-unused-parameters"),
        hyperparam("-a", "mega_lra_listop"),
        hyperparam("--task", "lra-text"),
        hyperparam("--input-type", "text"),
        hyperparam("--encoder-layers", 6),
        hyperparam("--n-dim", 16),
        hyperparam("--chunk-size", -1),
        hyperparam("--activation-fn", "silu"),
        hyperparam("--attention-activation-fn", "softmax"),
        hyperparam("--norm-type", "layernorm"),
        hyperparam("--sen-rep-type", "mp"),
        hyperparam("--criterion", "lra_cross_entropy"),
        hyperparam("--best-checkpoint-metric", "accuracy"),
        hyperparam("--maximize-best-checkpoint-metric"),
        hyperparam("--optimizer", "adam"),
        hyperparam("--lr", 0.001, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--adam-betas", "(0.9, 0.98)"),
        hyperparam("--adam-eps", 1e-8),
        hyperparam("--clip-norm", 1.0),
        hyperparam("--dropout", 0.1),
        hyperparam("--attention-dropout", 0.0),
        hyperparam("--act-dropout", 0.0),
        hyperparam("--weight-decay", 0.01),
        hyperparam("--batch-size", 8),
        hyperparam("--sentence-avg"),
        hyperparam("--update-freq", 1),
        hyperparam("--max-update", 90000),
        hyperparam("--max-sentences-valid", 256),
        hyperparam("--lr-scheduler", "linear_decay"),
        hyperparam("--total-num-update", 90000),
        hyperparam("--end-learning-rate", 0.0),
        hyperparam("--warmup-updates", 3000),
        hyperparam("--warmup-init-lr", 1e-07),
        hyperparam("--keep-last-epochs", 1),
        hyperparam("--required-batch-size-multiple", 1),
        #hyperparam("--save-dir", args.save_dir),
        hyperparam("--log-format", "simple"),
        hyperparam("--log-interval", 100),
        hyperparam("--num-workers", 0),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
