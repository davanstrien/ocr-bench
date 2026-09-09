#!/bin/sh
# Submit the reproducible experiment. Inference and all runtime checks run on Jobs.
set -eu
if [ "$#" -ne 3 ]; then
    echo "Usage: sh launch.sh smoke|full|score-smoke|score-full NAMESPACE/BUCKET RUN_ID" >&2
    exit 2
fi
phase=$1
bucket=$2
run_id=$3
source_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
image='ghcr.io/astral-sh/uv@sha256:85d4cb1afa769a7338e095b927bee941cf5ec92266c7424b3f6c0f2748567248'

submit() {
    flavor=$1
    timeout=$2
    name=$3
    shift 3
    hf jobs run --detach --name "$name" --label experiment=kraken-olmbench \
        --label "run=$run_id" --label "phase=$phase" \
        --flavor "$flavor" --timeout "$timeout" --secrets HF_TOKEN \
        --env PYTHONUNBUFFERED=1 --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=4 \
        --volume "$source_dir:/input:ro" \
        --volume "hf://buckets/$bucket:/bucket:rw" "$image" "$@"
}

case "$phase" in
    smoke)
        submit a10g-small 50m "kraken-$run_id-smoke" \
            sh /input/bootstrap.sh --mode smoke --run-id "$run_id"
        ;;
    full)
        for shard in 0 1 2 3; do
            submit a10g-small 3h "kraken-$run_id-shard-$shard" \
                sh /input/bootstrap.sh --mode full --run-id "$run_id" --shard "$shard"
        done
        ;;
    score-smoke|score-full)
        mode=${phase#score-}
        submit cpu-upgrade 1h "kraken-$run_id-score" \
            sh /input/score-bootstrap.sh --mode "$mode" --run-id "$run_id"
        ;;
    *)
        echo "Unknown phase: $phase" >&2
        exit 2
        ;;
esac
