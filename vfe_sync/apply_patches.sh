#!/bin/bash
# Apply Gauge-Transformer bug fixes to VFE-Transformer
#
# Usage: cd /path/to/VFE-Transformer && bash /path/to/apply_patches.sh
#
# Patches are applied in dependency order. If a patch fails to apply cleanly,
# it will be skipped with a warning. Review the output for MANUAL FIX NEEDED.

set -euo pipefail

PATCH_DIR="$(cd "$(dirname "$0")" && pwd)/patches"
APPLIED=0
FAILED=0
SKIPPED=0

echo "=== VFE-Transformer Sync: Applying Gauge-Transformer fixes ==="
echo "Patch directory: $PATCH_DIR"
echo "Target directory: $(pwd)"
echo ""

for patch in "$PATCH_DIR"/*.patch; do
    name=$(basename "$patch")

    # Check if patch has content
    if [ ! -s "$patch" ]; then
        echo "SKIP (empty): $name"
        ((SKIPPED++))
        continue
    fi

    # Check if patch applies cleanly
    if git apply --check "$patch" 2>/dev/null; then
        git apply "$patch"
        echo "  OK: $name"
        ((APPLIED++))
    else
        # Try with 3-way merge
        if git apply --check --3way "$patch" 2>/dev/null; then
            git apply --3way "$patch"
            echo "  OK (3-way): $name"
            ((APPLIED++))
        else
            echo "FAIL: $name — MANUAL FIX NEEDED"
            echo "      Run: git apply --reject $patch"
            echo "      Then fix the .rej files manually"
            ((FAILED++))
        fi
    fi
done

echo ""
echo "=== Summary ==="
echo "Applied: $APPLIED"
echo "Failed:  $FAILED"
echo "Skipped: $SKIPPED"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "Some patches failed. See SYNC_GUIDE.md for manual fix instructions."
    echo "For each failed patch, apply with --reject and fix conflicts:"
    echo "  git apply --reject <patch_file>"
    exit 1
fi

echo "All patches applied successfully!"
