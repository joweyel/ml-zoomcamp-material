# #!/usr/bin/python3
import os

HOME = os.getenv("HOME", "~/")
BIN_PATH = os.path.join(HOME, "bin")
KUBECTL_PATH = os.path.join(BIN_PATH, "kubectl")
KIND_PATH = os.path.join(BIN_PATH, "kind")

# Create bin-folder
if not os.path.exists(BIN_PATH):
    os.mkdir(BIN_PATH)

# Install kubectl
if not os.path.exists(KUBECTL_PATH):
    os.system(f"curl -Lo {KUBECTL_PATH} https://s3.us-west-2.amazonaws.com/amazon-eks/1.28.3/2023-11-14/bin/linux/amd64/kubectl")
else:
    print("kubectl already installed")

# Install kind
if not os.path.exists(KIND_PATH):
    # For AMD64 / x86_64
    os.system(f"[ $(uname -m) = x86_64 ] && curl -Lo {KIND_PATH} https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64")
    # For ARM64
    os.system(f"[ $(uname -m) = aarch64 ] && curl -Lo {KIND_PATH} https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-arm64")
    os.system(f"chmod +x {KIND_PATH}")
else:
    print("kind already installed")

