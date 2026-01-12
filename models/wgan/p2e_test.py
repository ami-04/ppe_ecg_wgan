# p2e_test.py
import os
import argparse
import numpy as np
import torch
from viewdata.data import get_data_loader
from saved_models.WGAN.models import GeneratorUNet
from utils import evaluate_generated_signal_quality  # optional, handle if not present

def load_generator(checkpoint_path, device):
    gen = GeneratorUNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # checkpoint might use keys like 'generator_state_dict'
    if 'generator_state_dict' in ckpt:
        gen.load_state_dict(ckpt['generator_state_dict'])
    else:
        # assume checkpoint is model.state_dict()
        gen.load_state_dict(ckpt)
    gen.eval()
    return gen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_prefix', required=True,
                        help="prefix path for your .npy datasets, e.g. data/your_dataset_")
    parser.add_argument('--checkpoint', required=True,
                        help="path to saved_models/<exp>/model_xxx.pth")
    parser.add_argument('--out_dir', default='test_results',
                        help="where to save generated signals and metrics")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--from_ppg', action='store_true', help='set if model converts from PPG->ECG')
    parser.add_argument('--compute_metrics', action='store_true',
                        help='If set, compute basic MSE per sample and try evaluate_generated_signal_quality')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # build dataloader (set shuffle_testing=False)
    _, val_loader = get_data_loader(
        args.dataset_prefix,
        batch_size=args.batch_size,
        from_ppg=args.from_ppg,
        shuffle_training=False,
        shuffle_testing=False
    )

    generator = load_generator(args.checkpoint, device)

    all_generated = []
    all_gt = []
    all_mse = []

    sample_idx = 0
    with torch.no_grad():
        for batch in val_loader:
            # batch may be (X,y) or (X,y,opeaks,rpeaks)
            if args.from_ppg:
                X, y, opeaks, rpeaks = batch
            else:
                X, y = batch

            X = X.to(device)
            y = y.to(device)

            gen_out = generator(X)
            # move to cpu numpy; remove channel dim
            gen_np = gen_out.detach().cpu().numpy()  # shape (B,1,L)
            y_np = y.detach().cpu().numpy()

            # save each sample in the batch
            for b in range(gen_np.shape[0]):
                gen_signal = np.squeeze(gen_np[b])  # (L,)
                gt_signal = np.squeeze(y_np[b])     # (L,)

                np.save(os.path.join(args.out_dir, f'gen_{sample_idx:05d}.npy'), gen_signal)
                np.save(os.path.join(args.out_dir, f'gt_{sample_idx:05d}.npy'), gt_signal)

                all_generated.append(gen_signal)
                all_gt.append(gt_signal)

                if args.compute_metrics:
                    mse = float(np.mean((gen_signal - gt_signal) ** 2))
                    all_mse.append(mse)

                sample_idx += 1

    all_generated = np.stack(all_generated, axis=0) if len(all_generated) else np.array([])
    all_gt = np.stack(all_gt, axis=0) if len(all_gt) else np.array([])

    # Save aggregated arrays for convenience
    if len(all_generated):
        np.save(os.path.join(args.out_dir, 'generated_all.npy'), all_generated)
        np.save(os.path.join(args.out_dir, 'gt_all.npy'), all_gt)

    # Basic metric reporting
    if args.compute_metrics and len(all_mse):
        mean_mse = float(np.mean(all_mse))
        print(f"Mean MSE over {len(all_mse)} samples: {mean_mse:.6f}")
        # save per-sample MSE
        np.save(os.path.join(args.out_dir, 'mse_per_sample.npy'), np.array(all_mse))

    # Optional: call repository's evaluation helper if available and desired
    try:
        if args.compute_metrics:
            print("Running repo's evaluate_generated_signal_quality (if implemented)...")
            # function signature may vary; adjust if necessary
            evaluate_generated_signal_quality(all_generated, all_gt, args.out_dir)
    except Exception as e:
        print("Skipping evaluate_generated_signal_quality due to error or missing implementation:", e)

    print(f"Saved {sample_idx} generated samples to {args.out_dir}")

if __name__ == '__main__':
    main()
