#!/usr/bin/env sh

if [ ! -f "serivce/rotational_invariance_pb2.py" ]; then
  SERVICE_DIR="."
  python -m grpc_tools.protoc -I proto \
    --python_out $SERVICE_DIR \
    --grpc_python_out $SERVICE_DIR \
    proto/rotational_invariance.proto
fi
