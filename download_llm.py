import os
import glob
from huggingface_hub import snapshot_download

# Перенаправляем системный кэш в папку проекта
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")

print("Скачивание всех частей модели Q4_K_M...")
snapshot_download(
    repo_id="IlyaGusev/saiga_llama3_8b_gguf", 
    allow_patterns=["*q4_K_M*.gguf", "*q4_k_m*.gguf", "*Q4_K_M*.gguf", "*q4_K*.gguf"], 
    local_dir="models"
)

# Находим скачанный файл и переименовываем, чтобы start.sh точно его нашел
found_files = glob.glob("models/*q4_*.gguf") + glob.glob("models/*Q4_*.gguf")
if found_files:
    target = "models/model-q4_K_M.gguf"
    if found_files[0] != target:
        os.rename(found_files[0], target)
    print("Модель успешно загружена!")
else:
    print("Ошибка: файл не найден в репозитории.")
