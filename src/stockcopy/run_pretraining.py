from pathlib import Path
from itertools import chain
import torch
import nnmodel
import pretraining
import training


# run training on gm games
def main():

    eposhs = 10
    batch_size = 64
    learning_rate = 0.001
    validatation_every = 10

    model = nnmodel.StockCopy().to('cuda')

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr = learning_rate,
        weight_decay = 0.0001
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max = eposhs
    )

    pgn_paths = [
        Path('data/Aronian.pgn'),
        Path('data/Carlsen.pgn'),
        Path('data/Caruana.pgn'),
        Path('data/Nakamura.pgn'),
        Path('data/So.pgn'),
        Path('data/VachierLagrave.pgn'),
    ]

    for eposh in range(eposhs):
        print(f'\neposh --- {eposh}')

        print(f'\nobtaining training samples...')

        # training samples
        train_samples = chain.from_iterable(
            pretraining.iterate_pgn_sample(pgn_path, 'train', validatation_every)
            for pgn_path in pgn_paths
        )

        print(f'\nobtainined training samples')

        train_batches = training.iterate_samples(train_samples, batch_size)

        print(f'\nbatched training samples\n')

        train_metrics = training.train_one_eposh(model, optimiser, train_batches)

        print(f'\ntrained eposh, obtaining val samples...')

        # validation samples
        val_samples = chain.from_iterable(
            pretraining.iterate_pgn_sample(pgn_path, 'val', validatation_every)
            for pgn_path in pgn_paths
        )

        print(f'\nobtainined validation samples')

        val_batches = training.iterate_samples(val_samples, batch_size)

        print(f'\nbatched validation samples\n')

        val_metrics = training.evaluate_one_eposh(model, val_batches)


        # update learning rate
        scheduler.step()


        # print output
        print(f'\n\neposh: {eposh}')

        print('\ntraining:')
        print(f'samples: {train_metrics['total_samples']}')
        print(f'policy loss: {train_metrics['policy_loss']:.4f}')
        print(f'value loss: {train_metrics['value_loss']:.4f}')
        print(f'total loss: {train_metrics['total_loss']:.4f}')

        print('\nvalidation:')
        print(f'samples: {val_metrics['total_samples']}')
        print(f'policy loss: {val_metrics['policy_loss']:.4f}')
        print(f'value loss: {val_metrics['value_loss']:.4f}')
        print(f'total loss: {val_metrics['total_loss']:.4f}')

        # save model
        torch.save(
            {
                'model_state_dict' : model.state_dict(),
                'optimiser_state_dict' : optimiser.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict()
            },
        'test_model.pt'
        )

main()