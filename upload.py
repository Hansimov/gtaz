import argparse
import json
import re
import shutil

from pathlib import Path
from typing import Literal
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
from tclogger import TCLogger, logstr

SECRETS_PATH = Path(__file__).parent / "secrets.json"
with open(SECRETS_PATH, "r", encoding="utf-8") as f:
    SECRETS = json.load(f)

HF_ENDPOINT = SECRETS.get("hf_endpoint", None)
HF_TOKEN = SECRETS["hf_token"]
REPO_ID = SECRETS["repo_id"]

BASE_DIR = Path(__file__).parent / "src" / "gtaz"
WEIGHT_EXTS = {".pth", ".onnx", ".engine"}
PROGRESS_FILE = Path(__file__).parent / "upload.json"

logger = TCLogger("HFUploader", use_prefix=True)


def log_src_dst(src: Path, dst: str):
    logger.mesg(f"*  src: {logstr.file(src)}")
    logger.mesg(f"-> dst: {logstr.file(dst)}")


class HFChunker:
    """Split large subdirectories into smaller chunks."""

    def __init__(self, max_files_per_chunk: int = 9000):
        global logger
        self.max_files_per_chunk = max_files_per_chunk
        logger = TCLogger("HFChunker", use_prefix=True)
        # Pattern: YYYY-MM-DD_HH-MM-SS-sss_FRAME.ext
        self.file_pattern = re.compile(
            r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})-\d{3}_(\d{4})\.(jpg|json)$"
        )

    def _parse_filename(self, filename: str) -> tuple[str, int, str] | None:
        """Parse filename and return (timestamp, frame_num, extension)."""
        match = self.file_pattern.match(filename)
        if match:
            timestamp = match.group(1)  # YYYY-MM-DD_HH-MM-SS
            frame_num = int(match.group(2))  # FRAME number
            extension = match.group(3)  # jpg or json
            return timestamp, frame_num, extension
        return None

    def _find_segment_starts(self, files: list[Path]) -> list[tuple[int, str]]:
        """Find files where frame number is 0001 (segment starts).

        Returns:
            List of (file_index, timestamp) tuples
        """
        segment_starts = []
        for idx, file_path in enumerate(files):
            parsed = self._parse_filename(file_path.name)
            if parsed:
                timestamp, frame_num, ext = parsed
                # Only consider .jpg files as markers, and frame_num == 1
                if ext == "jpg" and frame_num == 1:
                    segment_starts.append((idx, timestamp))
        return segment_starts

    def _calculate_chunks(
        self, segment_starts: list[tuple[int, str]], total_files: int
    ) -> list[tuple[int, int, str]]:
        """Calculate file chunks based on segment starts.

        Returns:
            List of (start_idx, end_idx, timestamp) tuples for each chunk
        """
        if not segment_starts:
            return []

        chunks = []
        current_start_idx = 0
        current_timestamp = segment_starts[0][1]
        current_count = 0

        for i in range(len(segment_starts)):
            segment_idx, segment_timestamp = segment_starts[i]

            # Calculate files count from current position to this segment
            if i == len(segment_starts) - 1:
                # Last segment: count to end of files
                segment_file_count = total_files - segment_idx
            else:
                # Count to next segment
                next_segment_idx = segment_starts[i + 1][0]
                segment_file_count = next_segment_idx - segment_idx

            # Check if adding this segment exceeds max
            if (
                current_count + segment_file_count > self.max_files_per_chunk
                and current_count > 0
            ):
                # Save current chunk
                chunks.append((current_start_idx, segment_idx, current_timestamp))
                # Start new chunk from this segment
                current_start_idx = segment_idx
                current_timestamp = segment_timestamp
                current_count = segment_file_count
            else:
                # Add to current chunk
                current_count += segment_file_count

        # Add final chunk
        chunks.append((current_start_idx, total_files, current_timestamp))

        return chunks

    def _move_files_to_new_dir(self, files_to_move: list[Path], target_dir: Path):
        """Move files to a new directory."""
        target_dir.mkdir(parents=True, exist_ok=True)
        for file_path in files_to_move:
            target_path = target_dir / file_path.name
            shutil.move(str(file_path), str(target_path))

    def chunk_subdirectory(self, sub_dir_path: Path):
        """Process a single subdirectory and split if necessary."""
        if not sub_dir_path.is_dir():
            logger.warn(f"Not a directory: {sub_dir_path}")
            return

        # Collect all files and sort by name
        all_files = sorted([f for f in sub_dir_path.iterdir() if f.is_file()])
        total_files = len(all_files)

        logger.note(f"Processing: {sub_dir_path.name}")
        logger.mesg(f"Total files: {total_files}")

        # Check if chunking is needed
        if total_files <= self.max_files_per_chunk:
            logger.okay(
                f"No chunking needed (files count {total_files} <= {self.max_files_per_chunk})"
            )
            return

        # Find segment starts
        segment_starts = self._find_segment_starts(all_files)
        if not segment_starts:
            logger.warn("No segment starts found (frame_0001), skipping")
            return

        logger.mesg(f"Found {len(segment_starts)} segments")

        # Calculate chunks
        chunks = self._calculate_chunks(segment_starts, total_files)
        logger.mesg(f"Will create {len(chunks)} chunks")

        # First chunk stays in original directory, move others
        for chunk_idx, (start_idx, end_idx, timestamp) in enumerate(chunks):
            chunk_files = all_files[start_idx:end_idx]
            chunk_size = len(chunk_files)

            if chunk_idx == 0:
                # Keep first chunk in original directory
                logger.okay(
                    f"Chunk 1/{len(chunks)}: Keep {chunk_size} files in {sub_dir_path.name}"
                )
            else:
                # Create new subdirectory for this chunk
                new_sub_dir = sub_dir_path.parent / f"{timestamp}"
                logger.mesg(
                    f"Chunk {chunk_idx + 1}/{len(chunks)}: Moving {chunk_size} files to {new_sub_dir.name}"
                )
                self._move_files_to_new_dir(chunk_files, new_sub_dir)
                logger.okay(
                    f"Chunk {chunk_idx + 1}/{len(chunks)}: Moved to {new_sub_dir.name}"
                )

    def chunk_cache_directory(self, cache_sub_dir: str):
        """Process all subdirectories in cache folder."""
        cache_dir = BASE_DIR / "cache" / cache_sub_dir
        if not cache_dir.exists():
            logger.error(f"Cache directory not found: {cache_dir}")
            return

        logger.note(f"Processing cache directory: {cache_dir}")

        # Get all subdirectories
        sub_dirs = sorted([d for d in cache_dir.iterdir() if d.is_dir()])
        logger.mesg(f"Found {len(sub_dirs)} subdirectories")

        for sub_dir in sub_dirs:
            self.chunk_subdirectory(sub_dir)

        logger.okay("All subdirectories processed!")


class HFDeleter:
    """Delete files and folders from Huggingface Hub repository."""

    def __init__(
        self,
        repo_id: str = REPO_ID,
        repo_type: Literal["model", "dataset", "space"] = "dataset",
    ):
        global logger
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.api = HfApi(endpoint=HF_ENDPOINT, token=HF_TOKEN)
        logger = TCLogger("HFDeleter", use_prefix=True)

    def delete_file(self, path_in_repo: str):
        """Delete a single file from the repository."""
        logger.note(f"Deleting file: {path_in_repo}")
        self.api.delete_file(
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
        )
        logger.okay(f"Deleted: {path_in_repo}")

    def delete_folder(self, repo_folder: str):
        """Delete an entire folder from the repository."""
        logger.note(f"Deleting repo folder: {repo_folder}")
        # List all files in the repository
        repo_files = self.api.list_repo_files(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
        )

        # Filter files that are in the target folder
        files_to_delete = [
            f for f in repo_files if f.startswith(repo_folder.rstrip("/") + "/")
        ]

        if not files_to_delete:
            logger.warn(f"No files found in repo folder: {repo_folder}")
            return

        logger.mesg(f"Found {len(files_to_delete)} files to delete")

        # Create delete operations
        operations = [
            CommitOperationDelete(path_in_repo=file_path)
            for file_path in files_to_delete
        ]

        # Execute deletion as a single commit
        self.api.create_commit(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            operations=operations,
            commit_message=f"Delete repo folder: {repo_folder}",
        )

        logger.okay(
            f"Deleted repo folder: {repo_folder} ({len(files_to_delete)} files)"
        )

    def delete_multiple_files(self, paths_in_repo: list[str]):
        """Delete multiple files in a single commit."""
        logger.note(f"Deleting {len(paths_in_repo)} files")

        operations = [
            CommitOperationDelete(path_in_repo=path) for path in paths_in_repo
        ]

        self.api.create_commit(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            operations=operations,
            commit_message=f"Delete {len(paths_in_repo)} files",
        )

        logger.okay(f"Deleted {len(paths_in_repo)} files")


class HFUploader:
    """Upload weights files and samples folders to Huggingface Hub."""

    def __init__(
        self,
        repo_id: str = REPO_ID,
        repo_type: Literal["model", "dataset", "space"] = "dataset",
    ):
        global logger
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.api = HfApi(endpoint=HF_ENDPOINT, token=HF_TOKEN)
        logger = TCLogger("HFUploader", use_prefix=True)

    def _get_weights_files(self, weights_dir: Path) -> list[Path]:
        weights = []
        for fp in weights_dir.iterdir():
            if fp.is_file() and fp.stem.endswith("_best") and fp.suffix in WEIGHT_EXTS:
                weights.append(fp)
        return weights

    def _upload_single_file(
        self, file_path: Path, path_in_repo: str, idx: int, total: int
    ):
        """Upload a single file and log the progress."""
        log_src_dst(file_path, path_in_repo)
        logger.mesg(f"[{idx}/{total}] Uploading: {file_path.name}")
        self.api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type=self.repo_type,
        )
        logger.okay(f"[{idx}/{total}] Uploaded: {file_path.name}")

    def upload_weights_files(self, sub_dir: str):
        """Upload weights files"""
        weights_dir = BASE_DIR / "checkpoints" / sub_dir
        weights = self._get_weights_files(weights_dir)
        total_weights = len(weights)

        logger.note(f"Uploading weights files:")
        logger.mesg(f"Total weights: {total_weights}")

        # load progress
        progress_key = f"{sub_dir}_weights"
        progress = self._load_progress(progress_key)
        uploaded_files = set(progress.get("uploaded_files", []))

        if uploaded_files:
            logger.hint(
                f"Resuming: {len(uploaded_files)}/{total_weights} already uploaded"
            )

        for idx, file_path in enumerate(weights, 1):
            if file_path.name in uploaded_files:
                continue

            path_in_repo = f"checkpoints/{sub_dir}/{file_path.name}"
            self._upload_single_file(file_path, path_in_repo, idx, total_weights)

            # Save progress after each successful upload
            uploaded_files.add(file_path.name)
            self._save_progress(
                progress_key,
                {
                    "total_files": total_weights,
                    "uploaded_files": list(uploaded_files),
                },
            )

        logger.okay(f"All weights uploaded!")

        # clear progress after completion
        # self._clear_progress(progress_key)

    def _load_progress(self, sub_dir: str) -> dict:
        """Load upload progress from JSON file."""
        if not PROGRESS_FILE.exists():
            return {}
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress = json.load(f)
        return progress.get(sub_dir, {})

    def _save_progress(self, sub_dir: str, progress_data: dict):
        """Save upload progress to JSON file."""
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                progresses = json.load(f)
        else:
            progresses = {}

        progresses[sub_dir] = progress_data

        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progresses, f, indent=2, ensure_ascii=False)

    def _clear_progress(self, sub_dir: str):
        """Clear progress for a specific sub_dir after completion."""
        if not PROGRESS_FILE.exists():
            return
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            progresses = json.load(f)
        if sub_dir in progresses:
            del progresses[sub_dir]
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(progresses, f, indent=2, ensure_ascii=False)

    def _collect_all_files(self, folder: Path) -> list[Path]:
        """Collect all files from a folder recursively."""
        all_files = list(folder.rglob("*"))
        return [f for f in all_files if f.is_file()]

    def _create_upload_operations(
        self, files: list[Path], base_dir: Path, path_in_repo_prefix: str
    ) -> list[CommitOperationAdd]:
        """Create CommitOperationAdd objects for a list of files."""
        operations = []
        for file_path in files:
            relative_path = file_path.relative_to(base_dir)
            file_path_in_repo = f"{path_in_repo_prefix}/{relative_path.as_posix()}"
            operations.append(
                CommitOperationAdd(
                    path_in_repo=file_path_in_repo,
                    path_or_fileobj=str(file_path),
                )
            )
        return operations

    def _upload_batch(
        self,
        operations: list[CommitOperationAdd],
        batch_num: int,
        total_batches: int,
        sub_dir: str,
    ):
        """Upload a batch of operations as a single commit."""
        self.api.create_commit(
            repo_id=self.repo_id,
            repo_type=self.repo_type,
            operations=operations,
            commit_message=f"Upload batch {batch_num}/{total_batches}",
        )
        logger.okay(f"Batch {batch_num}/{total_batches} uploaded")

        # save progress after successful upload
        self._save_progress(
            sub_dir,
            {
                "total_batches": total_batches,
                "completed_batches": batch_num,
            },
        )

    def upload_samples_folder(self, sub_dir: str, batch_size: int = 500):
        """Upload samples folder in batches."""
        samples_dir = BASE_DIR / "cache" / sub_dir
        path_in_repo = f"cache/{sub_dir}"
        logger.note(f"Uploading samples folder:")
        log_src_dst(samples_dir, path_in_repo)

        # load progress
        progress = self._load_progress(sub_dir)
        start_batch_num = progress.get("completed_batches", 0)

        # collect all files
        all_files = self._collect_all_files(samples_dir)
        total_files = len(all_files)
        logger.mesg(f"Total files: {total_files}")

        # upload in batches
        total_batches = (total_files + batch_size - 1) // batch_size

        if start_batch_num > 0:
            logger.hint(f"Resuming from batch {start_batch_num}/{total_batches}")

        for file_idx in range(start_batch_num * batch_size, total_files, batch_size):
            batch_files = all_files[file_idx : file_idx + batch_size]
            batch_num = file_idx // batch_size + 1
            logger.mesg(
                f"Uploading batch {batch_num}/{total_batches} ({len(batch_files)} files)"
            )
            operations = self._create_upload_operations(
                batch_files, samples_dir, path_in_repo
            )
            self._upload_batch(operations, batch_num, total_batches, sub_dir)

        # clear progress after completion
        # self._clear_progress(sub_dir)

        logger.okay(f"All batches uploaded!")

    def run(self, sub_dir: str):
        self.upload_weights_files(sub_dir)
        self.upload_samples_folder(sub_dir)


class HFArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(description="Huggingface Hub Uploader/Deleter/Chunker")
        self.add_argument(
            "-c",
            "--chunk",
            action="store_true",
            help="Chunk large subdirectories",
        )
        self.add_argument(
            "-k",
            "--chunk-cache-dir",
            type=str,
            help="Cache sub-directory name to chunk (e.g., 'agency_move')",
        )
        self.add_argument(
            "-d",
            "--delete",
            action="store_true",
            help="Delete from Huggingface Hub",
        )
        self.add_argument(
            "-r",
            "--delete-repo-folder",
            type=str,
            help="Repository folder path to delete (e.g., 'cache/agency_move')",
        )
        self.add_argument(
            "-u",
            "--upload",
            action="store_true",
            help="Upload to Huggingface Hub",
        )
        self.add_argument(
            "-s",
            "--upload-sub-dir",
            type=str,
            help="Sub-directory name for upload (e.g., 'agency_move')",
        )


def main():
    parser = HFArgParser()
    args = parser.parse_args()

    if args.chunk:
        if not args.chunk_cache_dir:
            parser.error("requires value of -k (--chunk-cache-dir) in chunk mode")
        chunker = HFChunker()
        chunker.chunk_cache_directory(args.chunk_cache_dir)

    if args.delete:
        if not args.delete_repo_folder:
            parser.error("requires value of -r (--delete-repo-folder) in delete mode")
        deleter = HFDeleter()
        deleter.delete_folder(args.delete_repo_folder)

    if args.upload:
        uploader = HFUploader()
        if not args.upload_sub_dir:
            parser.error("requires value of -s (--upload-sub-dir) in upload mode")
        uploader.run(args.upload_sub_dir)


if __name__ == "__main__":
    main()

    # Case: chunk large subdirectories
    # python upload.py -c -k agency_move

    # Case: delete a folder from repo
    # python upload.py -d -r "cache/agency_move"

    # Case: upload weights and samples
    # python upload.py -u -s agency_move
