FROM ubuntu:22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libomp-dev \
    g++-11 \
    && rm -rf /var/lib/apt/lists/*

# Set the default compiler to GCC 11
ENV CC=/usr/bin/gcc-11
ENV CXX=/usr/bin/g++-11

# Create build directory
WORKDIR /app

# Copy source code
COPY . .

# Build the library and executable
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_TESTING=OFF .. && \
    make -j$(nproc) && \
    make install

# Create a smaller runtime image
FROM ubuntu:22.04 AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libomp5 \
    && rm -rf /var/lib/apt/lists/*

# Copy the built executable from builder stage
COPY --from=builder /usr/local/bin/boostedpp /usr/local/bin/
COPY --from=builder /usr/local/lib/libboostedpp.* /usr/local/lib/

# Update dynamic linker
RUN ldconfig

# Set the working directory
WORKDIR /data

# Set the entrypoint
ENTRYPOINT ["boostedpp"]
CMD ["--help"]
