#!/bin/bash
# Train and evaluate the proposed model of the paper (configs/experiment/full_model.yaml).
#
#   scripts/run_full_model.sh train    # train into outputs/full_model/, then test that model
#   scripts/run_full_model.sh paper    # only test the pre-trained model in checkpoints/trained_model/
#
# Each test set (UMCU, MR-RATE) is written to <checkpoint>/test_<set>/.
#
# Environment variables:
#   QMAP_DATA_ROOT   folder with the preprocessed datasets (see README.md)
#   GPU              CUDA device (default 0)
#   PYTHON           interpreter (default: "uv run python")
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
PYTHON="${PYTHON:-uv run python}"

case "${1:-}" in
    train)
        $PYTHON train.py experiment=full_model "gpu=${GPU}"
        CHECKPOINT=trained ;;
    paper)
        CHECKPOINT=paper ;;
    *)
        echo "usage: $0 train|paper" >&2; exit 1 ;;
esac

for test_set in umcu mrrate; do
    $PYTHON test.py experiment=full_model "test_set=${test_set}" "checkpoint=${CHECKPOINT}" "gpu=${GPU}"
done
