# ==============================================================================
# STAGE 1: Base (Shared dependencies and environment)
# ==============================================================================
FROM quay.io/pypa/manylinux_2_28_x86_64:latest AS base

# 1. Configure repos and install ALL required tools once
RUN { \
    echo '[google-cloud-cli]'; \
    echo 'name=Google Cloud CLI'; \
    echo 'baseurl=https://packages.cloud.google.com/yum/repos/cloud-sdk-el8-x86_64'; \
    echo 'enabled=1'; \
    echo 'gpgcheck=1'; \
    echo 'repo_gpgcheck=0'; \
    echo 'gpgkey=https://packages.cloud.google.com/yum/doc/rpm-package-key.gpg'; \
} > /etc/yum.repos.d/google-cloud-sdk.repo && \
    dnf install -y dnf-plugins-core && \
    dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo && \
    dnf install -y \
    ccache \
    git \
    wget \
    unzip \
    ninja-build \
    google-cloud-cli \
    docker-ce-cli \
    docker-buildx-plugin && \
    dnf clean all && \
    rm -rf /var/cache/dnf

# 2. Set shared environment variables
# Added GDMR_BUILD_DIR to define the global build/cache path
# This flag must be set in all stages to ensure the cache is both
# populated and used.
ENV PATH="/usr/lib64/ccache:${PATH}" \
    CCACHE_DIR="/root/.cache/ccache" \
    CCACHE_NOHASHDIR="1" \
    CCACHE_BASEDIR="/" \
    PIP_NO_CACHE_DIR="1" \
    GDMR_BUILD_DIR="/opt/gdmr/build"

# ==============================================================================
# STAGE 2: Builder (Warms up the caches)
# ==============================================================================
FROM base AS builder

WORKDIR /tmp
COPY . .

# 3. Run build to populate /opt/gdmr/build/cache and /root/.cache/ccache
RUN python3.12 -m pip install subpackages/logging

# ==============================================================================
# STAGE 3: Final CI Runner
# ==============================================================================
FROM base AS final

# 4. Copy only the compiled caches from the builder stage
# Updated to drop the `/safari` namespace and copy the unified cache
COPY --from=builder /opt/gdmr/build/cache /opt/gdmr/build/cache
COPY --from=builder /root/.cache/ccache /root/.cache/ccache

WORKDIR /
CMD ["/bin/bash"]
