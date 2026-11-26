import torch
import torch.nn.functional as F

def train_batch(model, optimiser, scheduler, samples, batch_size):
    position_count = len(samples)

    model.train()

    optimiser.zero_grad(set_to_none = True)

    # shuffle training positions
    order = torch.randperm(position_count).tolist()

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_loss = 0

    # process mini-batches
    for start in range(0, position_count, batch_size):

        batch_indices = order[start:start + batch_size]

        batch = [
            samples[i]
            for i in batch_indices
        ]

        current_batch_size = len(batch)

        # prepare board inputs
        inputs = torch.stack([
            sample['board']
            for sample in batch
        ]).to('cuda')

        # find max number of legal moves
        max_legal_moves = max(
            len(sample['legal_ids'])
            for sample in batch
        )

        # prepare padded tensors on cpu
        legal_indices = torch.zeros(
            (current_batch_size, max_legal_moves),
            dtype = torch.long
        )

        legal_mask = torch.zeros(
            (current_batch_size, max_legal_moves),
            dtype = torch.bool
        )

        policy_targets = torch.zeros(
            (current_batch_size, max_legal_moves),
            dtype = torch.float32
        )

        value_targets = torch.stack(
            [
                sample['value_target']
                for sample in batch
            ]
        ).to(dtype = torch.float32, device = 'cuda')

        # fill variable length legal move data
        for i, sample in enumerate(batch):
            count = len(sample['legal_ids'])

            legal_indices[i, :count] = sample['legal_ids']

            legal_mask[i, :count] = True

            policy_targets[i, :count] = sample['policy_target']

        # transfer padded tensor to gpu
        legal_indices = legal_indices.to('cuda')
        legal_mask = legal_mask.to('cuda')
        policy_targets = policy_targets.to('cuda')

        # nn forward
        with torch.autocast(device_type = 'cuda', dtype = torch.bfloat16):
            policy_output, value_output = model(inputs)

        # flatten policy output
        policy_flat = policy_output.float().flatten(start_dim = 1)

        # select legal move logits
        legal_logits = policy_flat.gather(
            dim = 1,
            index = legal_indices
        )

        # mask padded moves
        legal_logits = legal_logits.masked_fill(
            ~legal_mask,
            float('-inf')
        )

        # policy probs
        log_probs = F.log_softmax(
            legal_logits,
            dim = 1
        )

        # avoid 0 * -inf
        safe_log_probs = log_probs.masked_fill(
            ~legal_mask,
            0
        )

        probs = log_probs.exp()



        # policy loss for each position
        policy_loss = -(policy_targets * safe_log_probs).sum(dim = 1).mean()
        
        # value loss
        value_loss = - (value_targets * F.log_softmax(value_output.float(), dim = 1)).sum(dim = 1).mean()

        # entropy
        entropy = -(probs * safe_log_probs).sum(dim = 1).mean()

        # combined loss
        loss = policy_loss + 0.5 * value_loss

        # accumulate gradients
        # weight by batch size so the final gradient represents the average over the whole game
        weighted_loss = (loss * current_batch_size / position_count)

        weighted_loss.backward()

        # record stats
        total_policy_loss += (policy_loss.detach().item() * current_batch_size)

        total_value_loss += (value_loss.detach().item() * current_batch_size)

        total_entropy += (entropy.detach().item() * current_batch_size)

        total_loss += (loss.detach().item() * current_batch_size)

    # update model once all mini batches
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm = 1
    )

    optimiser.step()
    scheduler.step()

    model.eval()

    return {
        'policy_loss' : total_policy_loss / position_count,
        'value_loss' : total_value_loss / position_count,
        'entropy' : total_entropy / position_count,
        'total_loss' : total_loss / position_count
    }
