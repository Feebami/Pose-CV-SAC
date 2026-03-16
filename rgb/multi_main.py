import subprocess

def main():
    """
    Runs the main training script for all configurations and a range of seeds.
    """
    for seed in range(1, 5):
        print(f"--- Running end-to-end training with seed: {seed} ---")
        command = ['python', '-m', 'rgb.main', '--seed', str(seed)]
        
        try:
            subprocess.run(command, check=True)
            print(f"--- Finished training for seed: {seed} ---\n")
        except subprocess.CalledProcessError as e:
            print(f"---!!! Training failed for seed: {seed} !!!---")
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
