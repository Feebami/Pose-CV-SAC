import subprocess

def main():
    """
    Runs the main training script for all configurations and a range of seeds.
    """
    for seed in range(1, 5):
        pose_command = ['python', '-m', 'pose.main', '--seed', str(seed)]
        e2e_command = ['python', '-m', 'end2end.main', '--seed', str(seed)]
        e2ept_command = ['python', '-m', 'end2end.main', '--seed', str(seed), '--pretrain']
        try:
            print(f"--- Running pose estimate training with seed: {seed} ---")
            subprocess.run(pose_command, check=True)
            print(f"--- Running end-to-end training with seed: {seed} ---")
            subprocess.run(e2e_command, check=True)
            print(f"--- Running end-to-end pretrain training with seed: {seed} ---")
            subprocess.run(e2ept_command, check=True)
            print(f"--- Finished training for seed: {seed} ---\n")
        except subprocess.CalledProcessError as e:
            print(f"---!!! Training failed for seed: {seed} !!!---")
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
