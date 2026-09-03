#!/usr/bin/env bash
# One entry point for every experiment. Settings live in experiments/<name>.conf;
# this resolves one, picks its venv, and either submits it or runs it inline.
#
#   ./run.sh <name>                     submit to SLURM
#   ./run.sh <name> --layer 9           submit, overriding a flag
#   ./run.sh --local <name> --sample 100    run inline (laptop / dev pod / debugging)
#   ./run.sh --dry-run <name>           print the resolved commands and stop
#   ./run.sh --after 12345 <name>       queue it to start only if job 12345 succeeds
#   ./run.sh --list                     list available experiments
#
# Overrides are appended to the config's ARGS. That works because argparse takes the
# LAST occurrence of a flag -- verified for plain store actions, store_true, and
# BooleanOptionalAction (so `--no-use-cache` after `--use-cache` gives False).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

EXP_DIR="experiments_configs"
TEMPLATE="slurms/_job.slurm"
RUN_LOG="$EXP_DIR/runs.tsv"

usage() { sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 1; }

list_experiments() {
    echo "Experiments in $EXP_DIR/:"
    for f in "$EXP_DIR"/*.conf; do
        [ -e "$f" ] || { echo "  (none yet)"; return; }
        printf '  %-34s %s\n' "$(basename "$f" .conf)" \
            "$(grep -m1 '^#' "$f" | sed 's/^# \?//')"
    done
}

local_run=0; dry_run=0; after=""
while [ $# -gt 0 ]; do
    case "$1" in
        --local)    local_run=1; shift ;;
        --dry-run)  dry_run=1;   shift ;;
        # afterok, not afterany: a validation chained behind a generation that crashed
        # would read a half-written parquet and record its failure as a low valid%.
        --after)    after="$2"; shift 2 ;;
        --list)     list_experiments; exit 0 ;;
        -h|--help)  usage ;;
        *) break ;;
    esac
done
[ $# -lt 1 ] && usage

name="$1"; shift
conf="$EXP_DIR/${name}.conf"
if [ ! -f "$conf" ]; then
    echo "No experiment '$name'. Looked for $conf" >&2
    list_experiments >&2
    exit 1
fi

# Config contract. SCRIPT and VENV are required; the rest have defaults that suit the
# CPU analysis jobs, which are the majority.
SCRIPT=""; VENV=""; ARGS=""
QUEUE="batch"; TIME="4:00:00"; MEM="32G"; CPUS="4"; GPUS="0"; EXCLUDE=""
# shellcheck disable=SC1090
source "$conf"

[ -n "$SCRIPT" ] || { echo "$conf: SCRIPT is required" >&2; exit 1; }
[ -n "$VENV" ]   || { echo "$conf: VENV is required (crystallm|relax|megnet)" >&2; exit 1; }

# The venv choice is a real footgun, so it is decided in exactly one place.
# crystallm_venv is a cu130 build and CANNOT run on the V100 (sm_70); anything touching
# M3GNet must use relax_venv. See CLAUDE.md.
case "$VENV" in
    crystallm) venv_path="CrystaLLM/crystallm_venv" ;;
    relax)     venv_path="relax_venv" ;;
    megnet)    venv_path="megnet_venv" ;;
    *) echo "$conf: VENV must be crystallm|relax|megnet, got '$VENV'" >&2; exit 1 ;;
esac
[ -d "$venv_path" ] || { echo "venv not found: $venv_path" >&2; exit 1; }

# SCRIPT may be a python file or, for the few experiments that are genuinely two calls
# with control flow between them, a shell snippet. A .py path gets `python` prepended.
case "$SCRIPT" in
    *.py) cmd="python $SCRIPT $ARGS $*" ;;
    *)    cmd="$SCRIPT $ARGS $*" ;;
esac
cmd="$(echo "$cmd" | tr '\n' ' ' | tr -s ' ' | sed 's/ *$//')"

# Job name: the config plus whatever was overridden, so eight concurrent runs are
# telling apart in squeue. Overrides are flattened, e.g. --layer 9 --width 2 -> l9-w2.
job_name="$name"
if [ $# -gt 0 ]; then
    # A path override (--input steering_results/.../foo.parquet) would otherwise put the
    # whole path in the name and squeue would show 40 identical prefixes. Keep the stem.
    parts=()
    for a in "$@"; do
        case "$a" in
            */*) parts+=("$(basename "$a" | sed 's/\.[a-z.]*$//')") ;;
            *)   parts+=("$a") ;;
        esac
    done
    suffix="$(printf '%s ' "${parts[@]}" | sed -E 's/--([a-z])[a-z-]*[= ]/\1/g; s/ +/-/g; s/-+$//')"
    job_name="${name}-${suffix}"
fi
job_name="${job_name:0:60}"

if [ "$dry_run" -eq 1 ]; then
    echo "config:   $conf"
    echo "venv:     $venv_path"
    echo "job name: $job_name"
    echo "command:  $cmd"
    if [ "$local_run" -eq 0 ]; then
        line="sbatch:   --job-name=$job_name --partition=$QUEUE --time=$TIME --mem=$MEM --cpus-per-task=$CPUS"
        [ -n "$after" ] && line="$line --dependency=afterok:$after"
        [ "${GPUS:-0}" != "0" ] && line="$line --gres=gpu:$GPUS"
        [ -n "$EXCLUDE" ] && line="$line --exclude=$EXCLUDE"
        echo "$line"
    fi
    exit 0
fi

export MATSTEER_VENV="$venv_path"
export MATSTEER_CMD="$cmd"

record_run() {   # timestamp, job id, config, git sha, resolved command
    mkdir -p "$EXP_DIR"
    [ -f "$RUN_LOG" ] || printf 'when\tjob_id\texperiment\tgit_sha\tcommand\n' > "$RUN_LOG"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$name" \
        "$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" "$cmd" >> "$RUN_LOG"
}

if [ "$local_run" -eq 1 ]; then
    record_run "local"
    echo "+ venv: $venv_path"
    echo "+ $cmd"
    # shellcheck disable=SC1090
    source "$venv_path/bin/activate"
    eval "$cmd"
    exit $?
fi

mkdir -p logs
sb=(--job-name="$job_name"
    --partition="$QUEUE" --time="$TIME" --mem="$MEM" --cpus-per-task="$CPUS"
    --output="logs/${name}_%j.out" --error="logs/${name}_%j.err"
    --export=ALL)
[ "${GPUS:-0}" != "0" ] && sb+=(--gres="gpu:${GPUS}")
[ -n "$EXCLUDE" ] && sb+=(--exclude="$EXCLUDE")
[ -n "$after" ] && sb+=(--dependency="afterok:$after")

job_id="$(sbatch --parsable "${sb[@]}" "$TEMPLATE")"
record_run "$job_id"
echo "$job_id  $job_name"
echo "  $cmd"
echo "  logs/${name}_${job_id}.out"
