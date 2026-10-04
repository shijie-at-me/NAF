"""The step loop of a training run: a budget of optimizer steps spread over epochs, and periodic checkpoints."""

from tqdm import tqdm


def step_budget(max_steps, epochs, num_batches):
    """Optimizer steps of a run: ``max_steps``, or fewer if ``epochs`` passes over the data run out first."""
    return min(max_steps, epochs * num_batches)


def training_batches(loader, epochs, total_steps, stop_after_first=False, on_epoch_end=None):
    """Yields ``(step, epoch, batch_idx, batch)`` until ``total_steps`` steps are done (one with
    ``stop_after_first``, for a sanity check). ``on_epoch_end()`` runs after every epoch, also the interrupted last.
    """
    step = 0
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            yield step, epoch, batch_idx, batch
            step += 1
            if step >= total_steps or stop_after_first:
                break
        if on_epoch_end is not None:
            on_epoch_end()
        if step >= total_steps or stop_after_first:
            return


def is_checkpoint_step(steps_done, total_steps, num_checkpoints=4):
    """Whether to save an intermediate checkpoint after ``steps_done`` steps: every ``total_steps //
    num_checkpoints`` steps, except at the end (the final checkpoint is saved anyway)."""
    interval = max(total_steps // num_checkpoints, 1)
    return steps_done < total_steps and steps_done % interval == 0
