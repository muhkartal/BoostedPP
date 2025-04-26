FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libomp-dev \
    libboost-all-dev \
    g++-11 \
    && rm -rf /var/lib/apt/lists/*

# Set the default compiler to GCC 11
ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11

# Install BoostedPP
WORKDIR /tmp/boostedpp
COPY .. /tmp/boostedpp
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_TESTING=OFF .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Build the API
WORKDIR /app
COPY . .
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && \
    cp boostedpp_api /usr/local/bin/

# Expose the port
EXPOSE 8080

# Set the working directory
WORKDIR /data

# Set the entrypoint
ENTRYPOINT ["boostedpp_api"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
