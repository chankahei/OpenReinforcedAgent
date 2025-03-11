import subprocess

with open('requirements.txt', 'r') as f:
    packages = f.readlines()

for package in packages:
    package = package.strip()
    if package: # ignore empty lines
        try:
            subprocess.check_call(['pip', 'install', package])
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}. Error: {e}")
            # Optionally log the failure to a file

print("Installation process completed (with potential failures).")