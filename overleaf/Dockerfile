# LaTeX Compiler Docker Image for Master Thesis
# This provides a complete TeX Live environment for compiling the thesis

FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install TeX Live and required packages
# Using texlive-latex-extra instead of texlive-full for faster builds
RUN apt-get update && apt-get install -y \
    biber \
    curl \
    git \
    latexmk \
    make \
    python3 \
    python3-pip \
    texlive-bibtex-extra \
    texlive-fonts-extra \
    texlive-fonts-recommended \
    texlive-lang-spanish \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-latex-recommended \
    texlive-science \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /thesis

# Set default command
CMD ["bash"]
