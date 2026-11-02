#!/usr/bin/env bash
set -euo pipefail

PANDWM_ROOT="${PANDWM_ROOT:-/path/to/pandwm_root}"
DATA_ROOT="$PANDWM_ROOT/data"
KEEP_TARS="${KEEP_TARS:-1}"

if [[ "$KEEP_TARS" != "0" && "$KEEP_TARS" != "1" ]]; then
    printf 'KEEP_TARS must be 0 or 1, got: %s\n' "$KEEP_TARS" >&2
    exit 1
fi

member_strip_count() {
    local member="$1"
    local normalized="${member#./}"
    local without_slashes="${normalized//\//}"
    printf '%s\n' "$(( ${#normalized} - ${#without_slashes} ))"
}

inspect_and_extract() {
    local archive="$1"
    local destination="$2"
    local extension="$3"
    local marker_dir="$destination/.extract-state"
    local marker
    marker="$marker_dir/$(basename -- "$archive").done"

    mkdir -p "$destination" "$marker_dir"

    if [[ -f "$marker" && ! "$archive" -nt "$marker" ]]; then
        local marker_probe
        marker_probe="$(sed -n '1p' "$marker")"
        if [[ -n "$marker_probe" && -f "$destination/$marker_probe" ]]; then
            printf 'Already extracted: %s\n' "$archive"
            if [[ "$KEEP_TARS" == "0" ]]; then
                rm -f -- "$archive"
            fi
            return 0
        fi
        printf 'Extraction marker needs revalidation: %s\n' "$archive"
    fi

    local listing
    listing="$(mktemp "${TMPDIR:-/tmp}/pandwm-train-tar-list.XXXXXX")"
    if ! tar -tf "$archive" >"$listing"; then
        rm -f -- "$listing"
        printf 'Cannot list archive (possibly incomplete): %s\n' \
            "$archive" >&2
        return 1
    fi

    printf '\nArchive: %s\n' "$archive"
    printf 'First members:\n'
    sed -n '1,8p' "$listing"

    if awk 'BEGIN { bad = 0 }
        /^\// { bad = 1 }
        /(^|\/)\.\.($|\/)/ { bad = 1 }
        END { exit bad ? 0 : 1 }
    ' "$listing"; then
        rm -f -- "$listing"
        printf 'Unsafe absolute or parent path in archive: %s\n' \
            "$archive" >&2
        return 1
    fi

    local member
    member="$(awk -v ext="$extension" '
        {
            path = $0
            sub(/^\.\//, "", path)
            count = split(path, parts, "/")
            base = parts[count]
            pattern = "^[0-9]+\\." ext "$"
            if (base ~ pattern) {
                print $0
                exit
            }
        }
    ' "$listing")"
    rm -f -- "$listing"

    if [[ -z "$member" ]]; then
        printf 'No timestamp-keyed .%s file found in %s\n' \
            "$extension" "$archive" >&2
        return 1
    fi

    local strip_count
    strip_count="$(member_strip_count "$member")"
    if [[ "$strip_count" == "0" ]]; then
        printf 'Layout: files are at archive root; strip-components=0\n'
    else
        printf 'Layout: files have %s leading component(s); strip-components=%s\n' \
            "$strip_count" "$strip_count"
    fi
    printf 'Layout probe member: %s\n' "$member"

    tar -xf "$archive" \
        -C "$destination" \
        --strip-components="$strip_count" \
        --no-same-owner \
        --no-same-permissions \
        --skip-old-files

    local probe_name
    probe_name="$(basename -- "$member")"
    if [[ ! -f "$destination/$probe_name" ]]; then
        printf 'Expected probe was not extracted: %s\n' \
            "$destination/$probe_name" >&2
        return 1
    fi

    printf '%s\n' "$probe_name" >"$marker"
    if [[ "$KEEP_TARS" == "0" ]]; then
        rm -f -- "$archive"
        printf 'Deleted successfully extracted archive: %s\n' "$archive"
    fi
}

extract_shards() {
    local source_dir="$1"
    local destination="$2"
    local extension="$3"
    local archives=()

    if [[ -d "$source_dir" ]]; then
        mapfile -d '' -t archives < <(
            find "$source_dir" -maxdepth 1 -type f -name '*.tar' \
                -print0 | sort -z
        )
    fi

    if (( ${#archives[@]} == 0 )); then
        if find "$destination" -maxdepth 1 -type f \
            -name "*.$extension" -print -quit 2>/dev/null | grep -q .; then
            printf 'No shard tars remain; using existing extraction: %s\n' \
                "$destination"
            return 0
        fi
        printf 'No .tar archives found in %s\n' "$source_dir" >&2
        exit 1
    fi

    local archive
    for archive in "${archives[@]}"; do
        inspect_and_extract "$archive" "$destination" "$extension"
    done
}

sanity_check_dir() {
    local label="$1"
    local directory="$2"
    local extension="$3"
    local count
    count="$(find "$directory" -maxdepth 1 -type f \
        -name "*.$extension" -printf '%f\n' | wc -l)"

    printf '\n%s: %s direct .%s files\n' "$label" "$count" "$extension"
    find "$directory" -maxdepth 1 -type f -name "*.$extension" \
        -printf '%f\n' | sort | sed -n '1,3p'

    if [[ "$count" == "0" ]]; then
        printf '%s extraction is empty: %s\n' "$label" "$directory" >&2
        exit 1
    fi

    local invalid
    invalid="$(find "$directory" -maxdepth 1 -type f \
        -name "*.$extension" -printf '%f\n' | \
        awk -v ext="$extension" \
            '$0 !~ ("^[0-9]+\\." ext "$") { print; exit }')"
    if [[ -n "$invalid" ]]; then
        printf '%s has a non-timestamp filename: %s\n' \
            "$label" "$invalid" >&2
        exit 1
    fi

    local nested
    nested="$(find "$directory" -mindepth 2 -type f \
        -name "*.$extension" -print -quit)"
    if [[ -n "$nested" ]]; then
        printf '%s contains a nested file; layout is not flat: %s\n' \
            "$label" "$nested" >&2
        exit 1
    fi
}

mkdir -p "$DATA_ROOT/train/rgb" "$DATA_ROOT/train/seg"

extract_shards \
    "$DATA_ROOT/train_train_tars" "$DATA_ROOT/train/rgb" jpg
extract_shards \
    "$DATA_ROOT/train_seg_tars" "$DATA_ROOT/train/seg" png

sanity_check_dir RGB "$DATA_ROOT/train/rgb" jpg
sanity_check_dir segmentation "$DATA_ROOT/train/seg" png

printf '\nTrain subset extraction complete. KEEP_TARS=%s\n' "$KEEP_TARS"
