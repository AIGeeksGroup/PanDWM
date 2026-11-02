#!/usr/bin/env bash
set -euo pipefail

SHARD_START="${SHARD_START:-0}"
SHARD_COUNT="${SHARD_COUNT:-20}"
DEPTH_URL_BASE="${DEPTH_URL_BASE:-https://hf-mirror.com/datasets/zeyuren2002/nuscene-360_train/resolve/main/depth}"
DEPTH_URL_BASE="${DEPTH_URL_BASE%/}"
DATA_ROOT="${DATA_ROOT:-/path/to/pandwm_root/data}"
KEEP_TARS="${KEEP_TARS:-0}"
MIN_FREE_BYTES=$((15 * 1024 * 1024 * 1024))
EXPECTED_FILES=1000

TAR_DIR="$DATA_ROOT/depth_tars"
DEPTH_DIR="$DATA_ROOT/train/depth"
MARKER_DIR="$DEPTH_DIR/.extract-state"
RGB_DIR="$DATA_ROOT/train/rgb"

require_nonnegative_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        printf '%s must be a nonnegative integer, got: %s\n' \
            "$name" "$value" >&2
        exit 1
    fi
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        printf '%s must be a positive integer, got: %s\n' \
            "$name" "$value" >&2
        exit 1
    fi
}

member_strip_count() {
    local member="$1"
    local normalized="${member#./}"
    local without_slashes="${normalized//\//}"
    printf '%s\n' "$(( ${#normalized} - ${#without_slashes} ))"
}

check_free_space() {
    local available
    available="$(df --output=avail -B1 "$DATA_ROOT" | awk 'NR == 2 {print $1}')"
    if [[ ! "$available" =~ ^[0-9]+$ ]]; then
        printf 'Could not determine free space for %s\n' "$DATA_ROOT" >&2
        exit 1
    fi
    printf 'Free space before download: %.2f GiB\n' \
        "$(awk -v bytes="$available" 'BEGIN {print bytes / 1073741824}')"
    if (( available < MIN_FREE_BYTES )); then
        printf 'ABORT: filesystem containing %s has less than 15 GiB free.\n' \
            "$DATA_ROOT" >&2
        exit 1
    fi
}

marker_is_valid() {
    local marker="$1"
    local archive="$2"
    local probe
    [[ -f "$marker" ]] || return 1
    if [[ -f "$archive" && "$archive" -nt "$marker" ]]; then
        return 1
    fi
    probe="$(sed -n '1p' "$marker")"
    [[ "$probe" =~ ^[0-9]{15,19}\.h5$ ]] || return 1
    [[ -f "$DEPTH_DIR/$probe" ]]
}

validate_archive_listing() {
    local archive="$1"
    local listing="$2"
    local timestamps="$3"
    local member strip_count duplicate

    if ! tar -tf "$archive" >"$listing"; then
        printf 'Cannot list archive (possibly incomplete): %s\n' \
            "$archive" >&2
        return 1
    fi

    if awk 'BEGIN {bad = 0}
        /^\// {bad = 1}
        /(^|\/)\.\.($|\/)/ {bad = 1}
        END {exit bad ? 0 : 1}
    ' "$listing"; then
        printf 'Unsafe absolute or parent path in archive: %s\n' \
            "$archive" >&2
        return 1
    fi

    member="$(awk '
        {
            path = $0
            sub(/^\.\//, "", path)
            count = split(path, parts, "/")
            base = parts[count]
            stem = base
            sub(/\.h5$/, "", stem)
            if (base ~ /^[0-9]+\.h5$/ &&
                length(stem) >= 15 && length(stem) <= 19) {
                print $0
                exit
            }
        }
    ' "$listing")"
    if [[ -z "$member" ]]; then
        printf 'No 15-19 digit timestamp-keyed .h5 file found in %s\n' \
            "$archive" >&2
        return 1
    fi

    strip_count="$(member_strip_count "$member")"
    if ! awk -v strip="$strip_count" -v output="$timestamps" '
        BEGIN {bad = 0; count = 0}
        {
            path = $0
            sub(/^\.\//, "", path)
            if (path == "" || path ~ /\/$/) {
                next
            }
            component_count = split(path, parts, "/")
            if (component_count <= strip) {
                print "Non-directory member has no file after stripping: " $0 \
                    > "/dev/stderr"
                bad = 1
                next
            }
            relative = parts[strip + 1]
            for (i = strip + 2; i <= component_count; i++) {
                relative = relative "/" parts[i]
            }
            stem = relative
            sub(/\.h5$/, "", stem)
            if (relative ~ /\// || relative !~ /^[0-9]+\.h5$/ ||
                length(stem) < 15 || length(stem) > 19) {
                print "Invalid post-strip depth member: " relative \
                    > "/dev/stderr"
                bad = 1
                next
            }
            print stem > output
            count++
        }
        END {
            close(output)
            if (count == 0) {
                print "Archive contains no depth files" > "/dev/stderr"
                bad = 1
            }
            exit bad ? 1 : 0
        }
    ' "$listing"; then
        return 1
    fi

    duplicate="$(sort "$timestamps" | uniq -d | sed -n '1p')"
    if [[ -n "$duplicate" ]]; then
        printf 'Archive contains duplicate depth timestamp: %s\n' \
            "$duplicate" >&2
        return 1
    fi
    sort -u -o "$timestamps" "$timestamps"
    printf 'Layout: strip-components=%s; probe member: %s\n' \
        "$strip_count" "$member"
    printf '%s\n' "$strip_count"
}

cross_check_rgb_shard_zero() {
    local depth_timestamps="$1"
    local rgb_timestamps="$2"
    local depth_count rgb_count overlap depth_only rgb_only
    local depth_first depth_last rgb_first rgb_last

    if [[ ! -d "$RGB_DIR" ]]; then
        printf 'ABORT: RGB directory is missing for shard-0 alignment: %s\n' \
            "$RGB_DIR" >&2
        exit 1
    fi
    find "$RGB_DIR" -maxdepth 1 -type f -name '*.jpg' -printf '%f\n' | \
        awk '
            /^[0-9]+\.jpg$/ {
                sub(/\.jpg$/, "")
                if (length($0) >= 15 && length($0) <= 19) print
            }
        ' | sort | sed -n '1,1000p' >"$rgb_timestamps"

    depth_count="$(wc -l <"$depth_timestamps")"
    rgb_count="$(wc -l <"$rgb_timestamps")"
    if (( rgb_count < EXPECTED_FILES )); then
        printf 'ABORT: RGB shard-0 range needs 1000 extracted timestamps; found %s.\n' \
            "$rgb_count" >&2
        exit 1
    fi

    depth_first="$(sed -n '1p' "$depth_timestamps")"
    depth_last="$(tail -n 1 "$depth_timestamps")"
    rgb_first="$(sed -n '1p' "$rgb_timestamps")"
    rgb_last="$(tail -n 1 "$rgb_timestamps")"
    overlap="$(comm -12 "$depth_timestamps" "$rgb_timestamps" | wc -l)"
    depth_only="$(comm -23 "$depth_timestamps" "$rgb_timestamps" | wc -l)"
    rgb_only="$(comm -13 "$depth_timestamps" "$rgb_timestamps" | wc -l)"

    printf 'Depth shard 0: count=%s first=%s last=%s\n' \
        "$depth_count" "$depth_first" "$depth_last"
    printf 'RGB shard 0:   count=%s first=%s last=%s\n' \
        "$rgb_count" "$rgb_first" "$rgb_last"
    printf 'Shard-0 exact timestamp overlap: %s\n' "$overlap"
    if (( overlap == 0 )); then
        printf 'ABORT: depth shard 0 and the extracted RGB shard-0 range have no timestamp overlap. The shard ordering is not aligned.\n' \
            >&2
        exit 1
    fi
    if (( depth_only != 0 && rgb_only != 0 )); then
        printf 'ABORT: shard-0 timestamps overlap, but neither timestamp set is a subset of the other (depth-only=%s, rgb-only=%s).\n' \
            "$depth_only" "$rgb_only" >&2
        exit 1
    fi
    printf 'Shard-0 depth/RGB timestamp sets are subset-compatible.\n'
}

verify_extracted_files() {
    local timestamps="$1"
    local stem
    while IFS= read -r stem; do
        if [[ ! -f "$DEPTH_DIR/$stem.h5" ]]; then
            printf 'Expected extracted depth file is missing: %s\n' \
                "$DEPTH_DIR/$stem.h5" >&2
            return 1
        fi
    done <"$timestamps"
}

extracted_bytes_for_list() {
    local timestamps="$1"
    awk 'NR == FNR {wanted[$1 ".h5"] = 1; next}
        $1 in wanted {total += $2; found++}
        END {
            if (found != length(wanted)) exit 1
            print total + 0
        }
    ' "$timestamps" <(
        find "$DEPTH_DIR" -maxdepth 1 -type f -name '*.h5' \
            -printf '%f %s\n'
    )
}

require_nonnegative_integer SHARD_START "$SHARD_START"
require_positive_integer SHARD_COUNT "$SHARD_COUNT"
if [[ "$KEEP_TARS" != "0" && "$KEEP_TARS" != "1" ]]; then
    printf 'KEEP_TARS must be 0 or 1, got: %s\n' "$KEEP_TARS" >&2
    exit 1
fi
if ! command -v aria2c >/dev/null 2>&1; then
    printf 'aria2c is required for resume-safe depth downloads.\n' >&2
    exit 1
fi

mkdir -p "$TAR_DIR" "$DEPTH_DIR" "$MARKER_DIR"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/pandwm-depth.XXXXXX")"
trap 'rm -rf -- "$work_dir"' EXIT

last_shard=$((SHARD_START + SHARD_COUNT - 1))
for ((shard_index = SHARD_START;
      shard_index <= last_shard;
      shard_index++)); do
    shard_name="$(printf 'depth-%05d.tar' "$shard_index")"
    archive="$TAR_DIR/$shard_name"
    marker="$MARKER_DIR/$shard_name.done"
    listing="$work_dir/$shard_name.list"
    timestamps="$work_dir/$shard_name.timestamps"

    printf '\nDepth shard %05d of %05d\n' "$shard_index" "$last_shard"
    if marker_is_valid "$marker" "$archive"; then
        printf 'Already extracted and marker validated: %s\n' "$marker"
        if [[ "$KEEP_TARS" == "0" && -f "$archive" ]]; then
            rm -f -- "$archive"
            printf 'Deleted retained archive: %s\n' "$archive"
        fi
        continue
    fi

    check_free_space
    aria2c -c -x1 -s1 -d "$TAR_DIR" -o "$shard_name" \
        "$DEPTH_URL_BASE/$shard_name"

    : >"$timestamps"
    validation_output="$(
        validate_archive_listing "$archive" "$listing" "$timestamps"
    )"
    strip_count="$(tail -n 1 <<<"$validation_output")"
    sed -n '1,$p' <<<"$validation_output" | sed '$d'

    file_count="$(wc -l <"$timestamps")"
    if (( file_count != EXPECTED_FILES )); then
        if (( shard_index == last_shard )); then
            printf 'WARNING: final requested shard has %s files; expected %s.\n' \
                "$file_count" "$EXPECTED_FILES" >&2
        else
            printf 'Archive %s has %s depth files; expected exactly %s.\n' \
                "$archive" "$file_count" "$EXPECTED_FILES" >&2
            exit 1
        fi
    fi

    if (( shard_index == 0 )); then
        cross_check_rgb_shard_zero \
            "$timestamps" "$work_dir/rgb-shard-00000.timestamps"
    fi

    tar -xf "$archive" \
        -C "$DEPTH_DIR" \
        --strip-components="$strip_count" \
        --no-same-owner \
        --no-same-permissions \
        --skip-old-files
    verify_extracted_files "$timestamps"

    if (( shard_index == SHARD_START )); then
        tar_bytes="$(stat -c '%s' "$archive")"
        extracted_bytes="$(extracted_bytes_for_list "$timestamps")"
        ratio="$(awk -v extracted="$extracted_bytes" -v tar="$tar_bytes" \
            'BEGIN {printf "%.6f", extracted / tar}')"
        printf 'First-shard extracted/tar size: %s/%s bytes (ratio %.3f)\n' \
            "$extracted_bytes" "$tar_bytes" "$ratio"
        if awk -v ratio="$ratio" 'BEGIN {exit ratio > 1.3 ? 0 : 1}'; then
            printf 'ABORT: first-shard extracted/tar ratio %.3f exceeds the 1.3 disk-budget limit. Archive retained and extraction left unmarked.\n' \
                "$ratio" >&2
            exit 1
        fi
    fi

    probe_name="$(sed -n '1p' "$timestamps").h5"
    printf '%s\n' "$probe_name" >"$marker.tmp.$$"
    mv -f -- "$marker.tmp.$$" "$marker"
    printf 'Extraction marker written: %s\n' "$marker"

    if [[ "$KEEP_TARS" == "0" ]]; then
        rm -f -- "$archive"
        printf 'Deleted successfully extracted archive: %s\n' "$archive"
    fi
done

printf '\nDepth subset extraction complete. KEEP_TARS=%s\n' "$KEEP_TARS"
