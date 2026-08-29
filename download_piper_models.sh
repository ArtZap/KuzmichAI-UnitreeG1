#!/bin/bash
# download_piper_models.sh
# Скачивает Piper-модели для всех языков из _WHISPER_TO_XTTS_LANG_MAP

BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"
DIR="$HOME/.local/share/piper"
mkdir -p "$DIR"

declare -A MODELS=(
    ["ru"]="ru/ru_RU/irina/medium/ru_RU-irina-medium"
    ["en"]="en/en_US/lessac/medium/en_US-lessac-medium"
    ["es"]="es/es_ES/mls_10246/medium/es_ES-mls_10246-medium"
    ["fr"]="fr/fr_FR/upmc/medium/fr_FR-upmc-medium"
    ["de"]="de/de_DE/thorsten/medium/de_DE-thorsten-medium"
    ["zh"]="zh/zh_CN/huayan/medium/zh_CN-huayan-medium"
    ["ar"]="ar/ar_JO/kareem/medium/ar_JO-kareem-medium"
    ["pt"]="pt/pt_BR/faber/medium/pt_BR-faber-medium"
    ["it"]="it/it_IT/riccardo/x_low/it_IT-riccardo-x_low"
    ["pl"]="pl/pl_PL/mls_6892/medium/pl_PL-mls_6892-medium"
    ["tr"]="tr/tr_TR/dfki/medium/tr_TR-dfki-medium"
    ["nl"]="nl/nl_NL/mls/medium/nl_NL-mls-medium"
    ["cs"]="cs/cs_CZ/jirka/medium/cs_CZ-jirka-medium"
    ["ja"]="ja/ja_JP/kenichi/medium/ja_JP-kenichi-medium"
    ["ko"]="ko/ko_KR/chunom/medium/ko_KR-chunom-medium"
    ["hu"]="hu/hu_HU/anna/medium/hu_HU-anna-medium"
)

for lang in "${!MODELS[@]}"; do
    path="${MODELS[$lang]}"
    name=$(basename "$path")
    echo "⬇️  Скачиваю $lang → $name ..."
    wget -q --show-progress -O "$DIR/${name}.onnx" "$BASE/${path}.onnx"
    wget -q -O "$DIR/${name}.onnx.json" "$BASE/${path}.onnx.json"
    echo "✅  $lang готов"
done

echo "🎉 Все модели скачаны в $DIR"
