import os
import argparse

from datasets.pong import PongState

def generate_pong_trajectory():
    pong_state = PongState()
    pong_dir = os.path.join('.', 'resources', 'pong')
    os.makedirs(pong_dir, exist_ok=True)
    print('Saving trajectory...')
    pong_state.save_trajectory(os.path.join(pong_dir, 't0.npy'), use_progress_bar=True)
    print('Animating trajectory...')
    pong_state.animate_trajectory(os.path.join(pong_dir, 't0.npy'), os.path.join(pong_dir, 't0.gif'), use_progress_bar=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generate-data', default=False, action='store_true',
        help='Generate a trajectory of data from the Pong dataset.'
    )
    args = parser.parse_args()
    
    if args.generate_data:
        generate_pong_trajectory()

if __name__ == '__main__':
    main()