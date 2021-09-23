import logging
import pickle
from collections import deque
from copy import deepcopy
from datetime import datetime
from os import makedirs
from os.path import join
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    checkpoint_path = "model_checkpoints"
    best_models_path = join(checkpoint_path, "best")
    run_id = f"two_states_in_{datetime.now():%d%m%Y_%H%M%S}"
    model_id = f"{run_id}"
    writer = SummaryWriter(comment=f"-{run_id}")

    makedirs(checkpoint_path, exist_ok=True)
    makedirs(best_models_path, exist_ok=True)

    device_token = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_token)

    n_rows, n_cols, n_to_win = 6, 7, 4

    if pretrained:
        load_checkpoint(pretrained_model_path, model, device=device)
        logger.info(f"Loaded pretrained model from \"{pretrained_model_path}\".")

    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2_regularization)
    scheduler = MultiStepLR(optimizer, milestones=milestones)

    curr_epoch_idx = 0
    curr_train_batch_idx = 0
    best_model_idx = 0
    n_total_train_games = 0
    n_total_eval_games = 0

    graceful_exit = GracefulExit()

    while graceful_exit.run:

        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        count_batches = 0

        model.train()
        for _ in range(train_steps):
            valid_start_idx = 0

            vals_t = torch.tensor([lst[-1].value for lst in batch_samples], device=device, dtype=torch.float32)
            probs_t = torch.tensor([lst[-1].probs for lst in batch_samples], device=device, dtype=torch.float32)

            log_probs_out, val_out = model(states_t)

            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO scheduler step

            batch_loss = loss.item()
            writer.add_scalar("batch/loss", batch_loss, curr_train_batch_idx)
            writer.add_scalar("batch/policy_loss", policy_loss.item(), curr_train_batch_idx)
            writer.add_scalar("batch/value_loss", value_loss.item(), curr_train_batch_idx)

            best_model_log = f"Best Model Idx: {best_model_idx - 1} " if best_model_idx > 0 else ""
            logger.info(f"Epoch {curr_epoch_idx}: Training - "
                        f"{best_model_log}Batch: {curr_train_batch_idx}: Loss {batch_loss}, "
                        f"(Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()})")

            curr_train_batch_idx += 1
            count_batches += 1

            epoch_loss += batch_loss
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()

        epoch_loss /= max(1.0, count_batches)
        epoch_policy_loss /= max(1.0, count_batches)
        epoch_value_loss /= max(1.0, count_batches)


        writer.add_scalar("epoch/loss", epoch_loss, curr_epoch_idx)
        writer.add_scalar("epoch/policy_loss", epoch_policy_loss, curr_epoch_idx)
        writer.add_scalar("epoch/value_loss", epoch_value_loss, curr_epoch_idx)
        logger.info(f"Epoch {curr_epoch_idx}: Loss: {epoch_loss}, "
                    f"Policy Loss: {epoch_policy_loss}, Value Loss: {epoch_value_loss}")

        curr_epoch_idx += 1


        pass
    pass


if __name__ == "__main__":
    main()
