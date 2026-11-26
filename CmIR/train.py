"""
Unified training / evaluation entry point for CmIR on CMU-MOSI and CMU-MOSEI.

The optimized objective is (paper Eq. 9):
    L = L_pred + lambda1 * R_inv + lambda2 * R_dec + lambda3 * R_rec

Example:
    python train.py --dataset mosi  (see README.md for the full configurations)
"""

import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import DebertaV2Tokenizer, get_linear_schedule_with_warmup

import global_configs
from global_configs import DEVICE
from model_cmir import CmIRForRegression

parser = argparse.ArgumentParser()
# Data / model paths
parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--data_path", type=str, default="datasets")
parser.add_argument("--max_seq_length", type=int, default=50)
# Batching / optimisation
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=128)
# Network dimensions
parser.add_argument("--inter_dim", type=int, default=256, help="hidden / MLP dimension")
parser.add_argument("--share_dim", type=int, default=150, help="shared representation dimension d")
parser.add_argument("--transformer_head", type=int, default=5)
parser.add_argument("--transformer_layer", type=int, default=3)
parser.add_argument("--kernel_size", type=int, default=3)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--drop_prob", type=float, default=0.3, help="attention dropout")
parser.add_argument("--dropout_unimodal", type=float, default=0.3)
# CmIR constraint weights (Eq. 9)
parser.add_argument("--lambda1", type=float, default=0.001, help="weight of R_inv")
parser.add_argument("--lambda2", type=float, default=0.01, help="weight of R_dec")
parser.add_argument("--lambda3", type=float, default=0.05, help="weight of R_rec")
parser.add_argument("--num_of_envs", type=int, default=1, help="number K of virtual environments")
parser.add_argument("--noise_scale", type=float, default=0.1, help="base noise coefficient alpha^(1)")
parser.add_argument("--weight_decouple", type=float, default=1.0, help="alpha in the orthogonality loss (Eq. 6)")
# OOD test-time corruption (Eq. 14; noise_type: 0 none, 1 Gaussian, 2 Laplace/erasing)
parser.add_argument("--noise_type", type=int, default=0)
parser.add_argument("--noise_experiment", type=int, default=0)
parser.add_argument("--percent", type=float, default=0.1, help="Laplace / erasing rate NR")
parser.add_argument("--percent2", type=float, default=0.1, help="Gaussian rate NR")
# Misc
parser.add_argument("--save_features", action="store_true", help="save extracted invariant features as .npy")
args = parser.parse_args()

global_configs.set_dataset_config(args.dataset)
ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM = (
    global_configs.ACOUSTIC_DIM,
    global_configs.VISUAL_DIM,
    global_configs.TEXT_DIM,
)


class InputFeatures:
    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id, sample_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sample_id = sample_id


def prepare_deberta_input(tokens, visual, acoustic, tokenizer):
    CLS, SEP = tokenizer.cls_token, tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)
    acoustic = np.concatenate((acoustic, np.zeros((pad_length, ACOUSTIC_DIM))))
    visual = np.concatenate((visual, np.zeros((pad_length, VISUAL_DIM))))
    input_ids += [0] * pad_length
    input_mask += [0] * pad_length
    segment_ids += [0] * pad_length

    return input_ids, visual, acoustic, input_mask, segment_ids


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []
    for example in examples:
        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        assert len(tokens) == len(inversions)
        aligned_visual = [visual[inv, :] for inv in inversions]
        aligned_audio = [acoustic[inv, :] for inv in inversions]
        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_deberta_input(
            tokens, visual, acoustic, tokenizer
        )

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert acoustic.shape[0] == max_seq_length
        assert visual.shape[0] == max_seq_length

        features.append(
            InputFeatures(input_ids, visual, acoustic, input_mask, segment_ids, label_id, segment)
        )
    return features


def get_appropriate_dataset(data, tokenizer):
    features = convert_to_features(data, args.max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_visual = torch.tensor(np.array([f.visual for f in features]), dtype=torch.float)
    all_acoustic = torch.tensor(np.array([f.acoustic for f in features]), dtype=torch.float)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    # Sample ids are placeholders kept for a stable 6-field dataset layout.
    all_sample_ids = torch.tensor(
        [float(i) for i in range(len(features))], dtype=torch.float
    )

    return TensorDataset(
        all_input_ids, all_visual, all_acoustic,
        all_label_ids, all_sample_ids, all_input_mask,
    )


def set_up_data_loader():
    with open(os.path.join(args.data_path, f"{args.dataset}.pkl"), "rb") as handle:
        data = pickle.load(handle)

    tokenizer = DebertaV2Tokenizer.from_pretrained(args.model)
    train_dataset = get_appropriate_dataset(data["train"], tokenizer)
    dev_dataset = get_appropriate_dataset(data["dev"], tokenizer)
    test_dataset = get_appropriate_dataset(data["test"], tokenizer)

    num_train_optimization_steps = (
        int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_step)
        * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=False
    )
    return train_dataloader, dev_dataloader, test_dataloader, num_train_optimization_steps


def set_random_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Seed:", seed)


def prep_for_training(num_train_optimization_steps):
    model = CmIRForRegression.from_pretrained(
        args.model, multimodal_config=args, num_labels=1,
    )
    model.to(DEVICE)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        num_training_steps=num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def _minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def train_epoch(model, dataloader, optimizer, scheduler):
    model.train()
    tr_loss, nb_steps = 0, 0
    history = {"inv": [], "mi": [], "rec": []}

    for batch in tqdm(dataloader, desc="train"):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, label_ids, _, input_mask = batch
        visual = _minmax(torch.squeeze(visual, 1))
        acoustic = _minmax(torch.squeeze(acoustic, 1))

        logits, loss_inv, loss_mi, loss_rec, _, _, _, _ = model(
            input_ids, visual, acoustic, label_ids, input_mask=input_mask
        )

        # Eq. 9: prediction loss plus the three CmIR constraints.
        loss = nn.MSELoss()(logits.view(-1), label_ids.view(-1))
        loss = loss + args.lambda1 * loss_inv + args.lambda2 * loss_mi + args.lambda3 * loss_rec

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step
        loss.backward()

        tr_loss += loss.item()
        nb_steps += 1
        history["inv"].append(loss_inv.item())
        history["mi"].append(loss_mi.item())
        history["rec"].append(loss_rec.item())

        if nb_steps % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return tr_loss / nb_steps, history


def eval_epoch(model, dataloader):
    model.eval()
    dev_loss, nb_steps = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids, _, _ = batch
            visual = _minmax(torch.squeeze(visual, 1))
            acoustic = _minmax(torch.squeeze(acoustic, 1))

            logits, *_ = model(input_ids, visual, acoustic, label_ids)
            loss = nn.MSELoss()(logits.view(-1), label_ids.view(-1))
            dev_loss += loss.item()
            nb_steps += 1
    return dev_loss / nb_steps


def test_epoch(model, dataloader):
    model.eval()
    preds, labels = [], []
    stored = {"l": [], "a": [], "v": [], "fusion": []}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="test"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids, _, _ = batch
            visual = _minmax(torch.squeeze(visual, 1))
            acoustic = _minmax(torch.squeeze(acoustic, 1))

            logits, _, _, _, z_l, z_a, z_v, fusion = model(
                input_ids, visual, acoustic, label_ids
            )
            stored["l"].append(z_l)
            stored["a"].append(z_a)
            stored["v"].append(z_v)
            stored["fusion"].append(fusion)

            preds.extend(np.squeeze(logits.detach().cpu().numpy()).tolist())
            labels.extend(np.squeeze(label_ids.detach().cpu().numpy()).tolist())

        if args.save_features:
            np.save(f"{args.dataset}_z_l_inv.npy", torch.cat(stored["l"], 0).cpu().numpy())
            np.save(f"{args.dataset}_z_a_inv.npy", torch.cat(stored["a"], 0).cpu().numpy())
            np.save(f"{args.dataset}_z_v_inv.npy", torch.cat(stored["v"], 0).cpu().numpy())
            np.save(f"{args.dataset}_fusion.npy", torch.cat(stored["fusion"], 0).cpu().numpy())
            np.save(f"{args.dataset}_labels.npy", np.array(labels))

    return np.array(preds), np.array(labels)


def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def test_score_model(model, dataloader, use_zero=False):
    preds, y_test = test_epoch(model, dataloader)
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])

    mult_a7 = multiclass_acc(
        np.clip(preds, -3, 3), np.clip(y_test, -3, 3)
    )
    mae_non = np.mean(np.abs(preds - y_test))
    corr_non = np.corrcoef(preds, y_test)[0][1]

    preds_nz, truth_nz = preds[non_zeros], y_test[non_zeros]
    mae = np.mean(np.abs(preds_nz - truth_nz))
    corr = np.corrcoef(preds_nz, truth_nz)[0][1]

    binary_pred = preds_nz >= 0
    binary_truth = truth_nz >= 0
    f_score = f1_score(binary_truth, binary_pred, average="weighted")
    acc = accuracy_score(binary_truth, binary_pred)

    return acc, mae, corr, f_score, mult_a7, mae_non, corr_non


def main():
    set_random_seed(args.seed)
    (
        train_loader, dev_loader, test_loader, num_steps,
    ) = set_up_data_loader()
    model, optimizer, scheduler = prep_for_training(num_steps)

    best_valid_loss = float("inf")
    best_metrics = None
    for epoch in range(args.n_epochs):
        train_loss, _ = train_epoch(model, train_loader, optimizer, scheduler)
        valid_loss = eval_epoch(model, dev_loader)
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")

        # Report test metrics every epoch; keep the checkpoint selected by validation.
        acc, mae, corr, f1, acc7, mae_non, corr_non = test_score_model(model, test_loader)
        print(
            f"TEST Acc2={acc:.4f} F1={f1:.4f} Acc7={acc7:.4f} "
            f"MAE={mae:.4f} Corr={corr:.4f} MAE_non={mae_non:.4f} Corr_non={corr_non:.4f}"
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_metrics = (acc, mae, corr, f1, acc7, mae_non, corr_non)
            print("New best validation model.")

    acc, mae, corr, f1, acc7, mae_non, corr_non = best_metrics
    print(
        "\n==== BEST (validation-selected) ====\n"
        f"Acc2={acc:.4f} F1={f1:.4f} Acc7={acc7:.4f} "
        f"MAE={mae:.4f} Corr={corr:.4f} MAE_non={mae_non:.4f} Corr_non={corr_non:.4f}"
    )


if __name__ == "__main__":
    main()
