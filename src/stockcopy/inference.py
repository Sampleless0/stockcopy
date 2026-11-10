import chess
import torch
import torch.nn.functional as F


# get a board and run the nn to get its policy and value outputs
def evaluate_policy_batch(model, input_batch, legal_indices, legal_mask):

    with torch.inference_mode():

        with torch.autocast(device_type = 'cuda', dtype = torch.bfloat16):

            policy_batch, value_batch = model(input_batch)

        # convert (8, 73, 8, 8) to (8, 4672)
        policy_flat = policy_batch.flatten(start_dim = 1)

        # get legal move logits eg [1.4, 2.3, 0.1, -1.3 etc]
        legal_logits = policy_flat.gather(dim = 1, index = legal_indices).float()

        # set non legal moves to negative infinity
        legal_logits = legal_logits.masked_fill(~legal_mask, float('-inf'))

        # get nn initial probabilities
        initial_prob_batch = F.softmax(legal_logits, dim = 1)
        
        # set to cpu
        initial_prob_batch = initial_prob_batch.cpu().tolist()


        # get wdl batch
        wdl_batch = F.softmax(value_batch.float(), dim = 1)

        # convert to 1 number (win - lose)
        value_batch = wdl_batch[:, 0] - wdl_batch[:, 2]

        # set to cpu
        value_batch = value_batch.cpu().tolist()

    return initial_prob_batch, value_batch