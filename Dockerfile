FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

# Set non-interactive mode to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install CMake
RUN apt-get update && \
    apt-get install -y cmake ca-certificates git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

