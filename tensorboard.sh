#!/usr/bin/env bash

echo "🚀 Starting TensorBoard..."
echo "📊 Monitoring logs in: logs/tensorboard"
echo ""
echo "Open your browser at: http://localhost:6006"
echo ""

tensorboard --logdir logs/tensorboard --port 6006 --bind_all
