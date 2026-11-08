"""Upload sched-sim Static Space to Hugging Face."""

from pathlib import Path
from huggingface_hub import HfApi, whoami


def main():
    api = HfApi()
    user_info = whoami()
    username = user_info["name"]
    print(f"Authenticated as Hugging Face user: {username} ({user_info.get('fullname')})")

    root = Path(__file__).resolve().parent.parent

    space_id = f"{username}/sched-sim"
    space_dir = root / "hf_space"

    if not space_dir.exists():
        raise FileNotFoundError(f"Directory {space_dir} does not exist. Run build_static_studio.py first!")

    print(f"\nCreating and uploading Static Space: {space_id}...")
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="static",
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=space_id,
        repo_type="space",
        commit_message="feat: publish sched-sim interactive Serving Scheduler & Gantt Studio v0.2.1",
    )
    print(f"\n✓ Space successfully deployed: https://huggingface.co/spaces/{space_id}")
    print("\n🎉 sched-sim Static Space is now live on Hugging Face!")


if __name__ == "__main__":
    main()
