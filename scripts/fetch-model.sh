#!/usr/bin/env bash
# Fetch model weights and test photos into models/. Large and gitignored, so fetch once per machine.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="${NAHPU_MODEL_DIR:-$ROOT/models}"

# name|url|sha256 — url pins an HF revision, not main, so the hash stays valid.
# The .jpg rows are NEON Biorepository photos (CC BY-SA 4.0) for tests and docs.
MODELS="
all-MiniLM-L6-v2.onnx|https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/onnx/model.onnx|6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452
all-MiniLM-L6-v2.tokenizer.json|https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/tokenizer.json|be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037
jina-clip-v1-text.onnx|https://huggingface.co/jinaai/jina-clip-v1/resolve/ceb3e44ca4d6eceaa4f3fb58b1c1a5748b3f29b6/onnx/text_model.onnx|d7b2e732e685dcaf5ca2a36e7f5d9169906859f4590766689dbf7b00b20da01b
jina-clip-v1-vision.onnx|https://huggingface.co/jinaai/jina-clip-v1/resolve/ceb3e44ca4d6eceaa4f3fb58b1c1a5748b3f29b6/onnx/vision_model.onnx|3032bf5f20bc39e23e9d2fff51ed1406341f1490bade128ffdcc6307af63c246
jina-clip-v1.tokenizer.json|https://huggingface.co/jinaai/jina-clip-v1/resolve/ceb3e44ca4d6eceaa4f3fb58b1c1a5748b3f29b6/tokenizer.json|b36ee0ed6d20d181de65ce729dea6169658a9cfa4dede6717ba9fa2e4fbd3bc7
fixture-neon-shrew.jpg|https://biorepo.neonscience.org/media/NEON_MAMC-VSS/00000/D00000153413-1_1701716781_lg.jpg|f320a3d961ede53a976672d6500faa733ca28ab0ce0394a88db54bedece235f3
fixture-neon-beetle.jpg|https://biorepo.neonscience.org/media/NEON_CARC-AP/OAES_/OAES_009.20160727.PASCAL.01.8_dorsal.jpg|c9f3adb94a80e40eca6d8193420a054a7de66c166ba6e06853c9eefdb9abadb7
nomic-embed-text-v1_5.onnx|https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/e9b6763023c676ca8431644204f50c2b100d9aab/onnx/model.onnx|147d5aa88c2101237358e17796cf3a227cead1ec304ec34b465bb08e9d952965
nomic-embed-text-v1_5.tokenizer.json|https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/e9b6763023c676ca8431644204f50c2b100d9aab/tokenizer.json|d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66
nomic-embed-vision-v1_5.onnx|https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5/resolve/e3a725bce72db07ca4adb1d83da08903f3ee02f8/onnx/model.onnx|53b4287e459193019bed6e094014344d4f3c892bd22b7159bf398d71eb82d750
nomic-embed-vision-v1_5.preprocessor_config.json|https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5/resolve/e3a725bce72db07ca4adb1d83da08903f3ee02f8/preprocessor_config.json|77436fccc0108364dd52185181d65cace7e830113e3b81b2bad8009a47f59b34
emb01_beetle_chlaenius_aestivus_dorsal.jpg|https://biorepo.neonscience.org/media/NEON_CARC-PV/NEONBET/NEON.BET.D08.002076_dorsal_1x.tiff.jpg|dfbcccb97266d6817f91792d5f0cf2eab4021e3f9a00512e42773affc30211f4
emb05_mosquito_aedes_monticola_lateral.jpg|https://biorepo.neonscience.org/media/NEON_MOSC-PV/MOS/MOS.D14.000124_lateral_2x_med.jpg|4c79712ba8cd77f7fc414f1f108e6b5a8dc3af59db62069a296cf0c2a26fef50
emb06_lizard_holbrookia_maculata_dorsal.jpg|https://biorepo.neonscience.org/media/NEON_HEVC-GBTS/A00000255/A00000255702-1_med.jpg|25ec021a49057adbe64b678077b8f22038549c7162b3bfd42ee63efc54f7e1f8
emb07_salamander_plethodon_cinereus_three.jpg|https://biorepo.neonscience.org/media/NEON_HEVC-GBAP/A00000328/A00000328482-2.jpg|1885d0796ba70395adbe13fa45f737c5df987d19941f6b3d1d33864298c068e3
emb08_toad_anaxyrus_americanus_two.jpg|https://biorepo.neonscience.org/media/NEON_HEVC-GBAP/00000/A00000360404-1_1713917340.jpg|d93d7ab00d057387b7dfe940e534a885537db28777927e0fd412b43a45b66b9d
emb10_mammal_peromyscus_ventral.jpg|https://biorepo.neonscience.org/media/NEON_MAMC-VSS/00000/D00000122950-3_1663177565.jpg|0244170d4b78ee06fb77550408367239bbe3f70427d116e3b6acf8ab3f196f31
PP-OCRv6_small_det.onnx|https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/resolve/28fe5895c24fd108c19eb3e8479f4ab385fbfc62/inference.onnx|d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e
PP-OCRv6_small_det.yml|https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx/resolve/28fe5895c24fd108c19eb3e8479f4ab385fbfc62/inference.yml|193f435274bf9f0b5f71a929bbfbcf148282df7e633b34e7c373e8f44741b516
PP-OCRv6_small_rec.onnx|https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/b8f84f0b80c529de40b4fbb3544b84fa7233a513/inference.onnx|5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634
PP-OCRv6_small_rec.yml|https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/b8f84f0b80c529de40b4fbb3544b84fa7233a513/inference.yml|ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1
ocr01_beetle_pasimachus_sublaevis_label_closeup.jpg|https://biorepo.neonscience.org/media/CMNH_INV/NEONBET/NEON.BET.D03.001188_med.jpg|528c0b902003ec66fbc2617edc52ff8183d405e6f07dd7f03c9e35aac732427e
ocr07_mammal_blarina_brevicauda_handwritten_tag.jpg|https://biorepo.neonscience.org/media/NEON_MAMC-VSS/00000/B00000031028-1_1624657069.jpg|1aafff6e32bd509097d4ed437f86b634031ea84988afd0ff82edf0ddd511a92b
ocr08_mammal_sorex_trowbridgii_handwritten_notes.jpg|https://biorepo.neonscience.org/media/NEON_MAMC-VSS/00000/D00000028740-2_lg.jpg|e0b5a8e219169bf6da7e1b9453e5682155ece492ddd81ff92b65573dde65ccc7
ocr11_plant_hieracium_scouleri_herbarium_printed_label.jpg|https://biorepo.neonscience.org/media/NEON_TPLC-HV/B00000053/B00000053749.jpg|053a2be66c1998b8dee8f36ae21b80f2e80ee7122fe865f8ec9914e76dfd91e5
ocr02_beetle_pasimachus_sublaevis_label_sideways.jpg|https://biorepo.neonscience.org/media/CMNH_INV/NEONBET/NEON.BET.D03.001210_med.jpg|a8cd96d9b3e77f184d287f15cea4060cff790857e50730ac65ad93818fc84345
ocr03_beetle_pasimachus_marginatus_label_occluded.jpg|https://biorepo.neonscience.org/media/CMNH_INV/NEONBET/NEON.BET.D03.000206_med.jpg|ceebf1182130d2a235e11964b1098e2b16bde6449ca9a65923dd7d8bddd18296
hard01_mammal_napaeozapus_skull_ink_number.jpg|https://biorepo.neonscience.org/media/NEON_MAMC-VOS/00000/skull_B00000061067-1_1648852819.jpg|1bdc907c3f60d161664a2136c07ca71e6864a0adcfd952d44f6bc5e5a8a4d199
hard02_mammal_sorex_rohweri_skull_ink_number.jpg|https://biorepo.neonscience.org/media/NEON_MAMC-VGBT/00000/D00000166461-10_1690845562.jpg|4e88112399f10c51db9098551691b07008b80c196019d50fc038d41e7384ff1a
"

print_usage() {
    echo "usage: $0 [model-name ...]    (default: all)"
    echo
    echo "A name cut at its first dot fetches every file with exactly that stem:"
    echo "nomic-embed-text-v1_5 fetches its .onnx and its .tokenizer.json."
    echo
    echo "available:"
    list_model_names
}

list_model_names() {
    echo "$MODELS" | while IFS='|' read -r name _ _; do
        [ -n "$name" ] && echo "  $name"
        true
    done
}

checksum_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

download_to() {
    local url="$1" dest="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fSL --retry 3 -o "$dest.part" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$dest.part" "$url"
    else
        echo "error: need curl or wget" >&2
        exit 1
    fi
    mv "$dest.part" "$dest"
}

verify_or_report() {
    local name="$1" dest="$2" expected="$3"
    local actual
    actual="$(checksum_of "$dest")"

    if [ -z "$actual" ]; then
        echo "  warning: no sha256 tool, skipping verification" >&2
    elif [ "$expected" = "-" ]; then
        echo "  sha256 $actual"
        echo "  (not pinned — paste this into scripts/fetch-model.sh)"
    elif [ "$actual" != "$expected" ]; then
        echo "  error: sha256 mismatch for $name" >&2
        echo "    expected $expected" >&2
        echo "    actual   $actual" >&2
        exit 1
    else
        echo "  sha256 ok"
    fi
}

fetch_model() {
    local name="$1" url="$2" expected="$3"
    local dest="$DEST_DIR/$name"

    if [ -f "$dest" ]; then
        echo "already present: $name"
    else
        echo "downloading $name"
        mkdir -p "$DEST_DIR"
        download_to "$url" "$dest"
    fi

    verify_or_report "$name" "$dest" "$expected"
}

case "${1:-}" in
    -h|--help) print_usage; exit 0 ;;
esac

# Space-padded so a substring test works on the bash 3.2 that macOS ships.
REQUESTED=" $* "

echo "$MODELS" | while IFS='|' read -r name url sha; do
    [ -z "$name" ] && continue
    model="${name%%.*}"
    if [ -n "$*" ] && [ "${REQUESTED#* $name }" = "$REQUESTED" ] \
        && [ "${REQUESTED#* $model }" = "$REQUESTED" ]; then
        continue
    fi
    fetch_model "$name" "$url" "$sha"
done

echo "models in $DEST_DIR"
