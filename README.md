# wan-deploy

A simple, elegant deployment utility for WAN environments.

## Features
- Lightweight and easy to use
- Designed for clean code and maintainability

## Setup & Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/calemcnulty-gai/wan-deploy.git
   cd wan-deploy
   ```
2. (Optional) Create and activate a virtual environment:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies (if requirements.txt is added):
   ```sh
   pip install -r requirements.txt
   ```

## Usage

- Main scripts:
  - `server.py`: Start the deployment server
  - `worker.py`: Run a deployment worker
  - `generate_video.py`: Utility for video generation

Example:
```sh
python server.py
```

## Testing

Testing instructions will be added as the project evolves.

## Contributing

PRs and issues welcome!

## Environment

This project is being run on an EC2 instance (i-0c96c24927baa7f38), which is a g5.12xlarge built on the Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6 (Ubuntu 22.04) with AMI IDs ami-0fcdcdcc9cf0407ae (x86) and ami-00d7a8764c5623acd (Arm). NVIDIA drivers have been added to the instance. 