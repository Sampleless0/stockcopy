import torch
import torch.nn.functional as F

def combine_samples(samples):

    # get batch size
    batch_size = len(samples)

    # stack board positions
    boards = torch.stack([
        sample['board']
        for sample in samples
    ])

    # get max legal moves
    max_legal_moves = max([
        len(sample['legal_indices'])
        for sample in samples
    ])

    # make legal tensors
    legal_indices = torch.zeros((batch_size, max_legal_moves), dtype = torch.long)

    legal_mask = torch.zeros((batch_size, max_legal_moves), dtype = torch.bool)

    policy_targets = torch.zeros((batch_size, max_legal_moves), dtype = torch.float32)

    value_targets = torch.tensor(
        [
            sample['value_target']
            for sample in samples
        ],
        dtype = torch.float32
    )

    # fill values
    for i, sample in enumerate(samples):

        count = len(sample['legal_indices'])

        legal_indices[i, :count] = sample['legal_indices']

        legal_mask[i, :count] = True

        policy_targets[i, :count] = sample['policy_target']

    return {
        'boards' : boards,
        'legal_indices' : legal_indices,
        'legal_masks' : legal_mask,
        'policy_targets' : policy_targets,
        'value_targets' : value_targets
    }



def iterate_samples(samples, batch_size):

    batch = []

    for sample in samples:

        batch.append(sample)

        if len(batch) == batch_size:
            yield combine_samples(batch)
            batch = []

    # excess samples
    if batch:
        yield combine_samples(batch)


def train_one_eposh(model, optimiser, batches):

    model.train()

    total_policy_loss = 0
    total_value_loss = 0
    total_loss = 0
    total_samples = 0

    for i, batch in enumerate(batches):
        print(f'\rtraining batch {i+1} ', end = '', flush = True)

        # transfer to gpu
        input_batch = batch['boards'].to('cuda')

        legal_indices = batch['legal_indices'].to('cuda')
        legal_mask = batch['legal_masks'].to('cuda')

        policy_targets = batch['policy_targets'].to('cuda')
        value_targets = batch['value_targets'].to('cuda')

        current_batch_size = input_batch.shape[0]

        optimiser.zero_grad(set_to_none = True)

        # forward
        with torch.autocast(device_type = 'cuda', dtype = torch.bfloat16):

            policy_batch, value_batch = model(input_batch)

            # flatten output
            policy_flat = policy_batch.flatten(start_dim = 1)

            legal_logits = policy_flat.gather(dim = 1, index = legal_indices).float()

            # remove invalid moves
            legal_logits = legal_logits.masked_fill(~legal_mask, float('-inf'))

            # calc policy loss
            log_probs = F.log_softmax(legal_logits, dim = 1)

            safe_log_probs = log_probs.masked_fill(~legal_mask, 0)

            policy_losses = -(policy_targets * safe_log_probs).sum(dim = 1)

            policy_loss = policy_losses.mean()

            # convert wdl values into labels
            value_classes = (1 - value_targets).round().long()

            # calc value loss
            value_loss = F.cross_entropy(
                value_batch.float(),
                value_classes
            )

            # combined loss
            loss = policy_loss + (0.5 * value_loss)

            # backpropagate
            loss.backward()

            # prevent extreme gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm = 1
            )

            # update model
            optimiser.step()

            # update stats
            total_policy_loss += policy_loss.item() * current_batch_size
            total_value_loss += value_loss.item() * current_batch_size
            total_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

    return {
        'policy_loss' : total_policy_loss / total_samples,
        'value_loss' : total_value_loss / total_samples,
        'total_loss' : total_loss / total_samples,
        'total_samples' : total_samples
    }

def evaluate_one_eposh(model, batches):

    model.eval()

    total_policy_loss = 0
    total_value_loss = 0
    total_loss = 0
    total_samples = 0

    with torch.inference_mode():

        for i, batch in enumerate(batches):
            print(f'\revaluating batch {i+1} ', end = '', flush = True)

            # transfer to gpu
            input_batch = batch['boards'].to('cuda')

            legal_indices = batch['legal_indices'].to('cuda')
            legal_mask = batch['legal_masks'].to('cuda')

            policy_targets = batch['policy_targets'].to('cuda')
            value_targets = batch['value_targets'].to('cuda')

            current_batch_size = input_batch.shape[0]


            # forward
            with torch.autocast(device_type = 'cuda', dtype = torch.bfloat16):

                policy_batch, value_batch = model(input_batch)

                # flatten output
                policy_flat = policy_batch.flatten(start_dim = 1)

                legal_logits = policy_flat.gather(dim = 1, index = legal_indices).float()

                # remove invalid moves
                legal_logits = legal_logits.masked_fill(~legal_mask, float('-inf'))

                # calc policy loss
                log_probs = F.log_softmax(legal_logits, dim = 1)

                safe_log_probs = log_probs.masked_fill(~legal_mask, 0)

                policy_losses = -(policy_targets * safe_log_probs).sum(dim = 1)

                policy_loss = policy_losses.mean()

                # convert wdl values into labels
                value_classes = (1 - value_targets).round().long()

                # calc value loss
                value_loss = F.cross_entropy(
                    value_batch.float(),
                    value_classes
                )

                # combined loss
                loss = policy_loss + (0.5 * value_loss)

                # update stats
                total_policy_loss += policy_loss.item() * current_batch_size
                total_value_loss += value_loss.item() * current_batch_size
                total_loss += loss.item() * current_batch_size
                total_samples += current_batch_size

    return {
        'policy_loss' : total_policy_loss / total_samples,
        'value_loss' : total_value_loss / total_samples,
        'total_loss' : total_loss / total_samples,
        'total_samples' : total_samples
    }