import subprocess

def main():
    """
    Runs the main training script for all configurations and a range of seeds.
    """
    for seed in range(0, 5):
        print(f"--- Running pose estimate training with seed: {seed} ---")
        command = ['python', '-m', 'pose.main', '--seed', str(seed)]
        
        try:
            subprocess.run(command, check=True)
            print(f"--- Finished training for seed: {seed} ---\n")
        except subprocess.CalledProcessError as e:
            print(f"---!!! Training failed for seed: {seed} !!!---")
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
