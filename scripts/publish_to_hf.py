"""Upload sched-sim serving traces dataset to Hugging Face."""

from pathlib import Path

from huggingface_hub import HfApi, whoami


def main():
    api = HfApi()
    user_info = whoami()
    username = user_info["name"]
    print(
        f"Authenticated as Hugging Face user: {username} ({user_info.get('fullname')})"
    )

    root = Path(__file__).resolve().parent.parent

    dataset_id = f"{username}/llm-serving-traces"
    dataset_dir = root / "hf_dataset"

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Directory {dataset_dir} does not exist. Run generate_traces_dataset.py first!"
        )

    print(f"\nCreating and uploading Dataset: {dataset_id}...")
    api.create_repo(
        repo_id=dataset_id,
        repo_type="dataset",
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(dataset_dir),
        repo_id=dataset_id,
        repo_type="dataset",
        commit_message="feat: upload LLM serving traces and scheduler performance benchmark dataset",
    )
    print(
        f"\n✓ Dataset successfully deployed: https://huggingface.co/datasets/{dataset_id}"
    )
    print("\n🎉 sched-sim serving traces dataset is now live on Hugging Face!")


if __name__ == "__main__":
    main()
