# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tao_sdk.script_runner."""

import json
import os

import pytest
from unittest.mock import MagicMock, patch, mock_open, call

from tao_sdk.script_runner import (
    _is_remote,
    _get_nested,
    _set_nested,
    _s3_to_lustre,
    _is_archive,
    _strip_archive_suffix,
    download_file,
    download_folder,
    download_selective,
    extract_archive,
    extract_archives_in_dir,
    upload_results,
    run,
)


class TestHelpers:

    def test_is_remote_s3(self):
        assert _is_remote("s3://bucket/path") is True

    def test_is_remote_local(self):
        assert _is_remote("/local/path") is False

    def test_is_remote_relative(self):
        assert _is_remote("relative/path") is False

    def test_get_nested_simple(self):
        d = {"a": {"b": {"c": 42}}}
        assert _get_nested(d, "a.b.c") == 42

    def test_get_nested_missing(self):
        d = {"a": {"b": 1}}
        assert _get_nested(d, "a.c") is None

    def test_get_nested_top_level(self):
        d = {"results_dir": "/results/123"}
        assert _get_nested(d, "results_dir") == "/results/123"

    def test_set_nested_creates_path(self):
        d = {}
        _set_nested(d, "a.b.c", 42)
        assert d == {"a": {"b": {"c": 42}}}

    def test_set_nested_overwrites(self):
        d = {"a": {"b": {"c": 1}}}
        _set_nested(d, "a.b.c", 99)
        assert d["a"]["b"]["c"] == 99

    def test_set_nested_top_level(self):
        d = {"x": 1}
        _set_nested(d, "x", 2)
        assert d["x"] == 2


class TestDownloadFile:

    @patch("tao_sdk.script_runner.fsspec")
    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    @patch("builtins.open", mock_open())
    @patch("shutil.copyfileobj")
    def test_downloads_file(self, mock_copy, mock_makedirs, mock_exists, mock_fsspec):
        mock_src = MagicMock()
        mock_fsspec.open.return_value.__enter__ = MagicMock(return_value=mock_src)
        mock_fsspec.open.return_value.__exit__ = MagicMock(return_value=False)

        download_file("s3://bucket/data.json", "/local/data.json")

        mock_fsspec.open.assert_called_once_with("s3://bucket/data.json", "rb")
        mock_copy.assert_called_once()

    @patch("tao_sdk.script_runner.fsspec")
    @patch("os.path.exists", return_value=True)
    @patch("os.makedirs")
    def test_skips_existing_file(self, mock_makedirs, mock_exists, mock_fsspec):
        """No download if local file already exists."""
        download_file("s3://bucket/data.json", "/local/data.json")
        mock_fsspec.open.assert_not_called()


class TestDownloadFolder:

    @patch("tao_sdk.script_runner.download_file")
    @patch("tao_sdk.script_runner.fsspec")
    def test_downloads_all_files(self, mock_fsspec, mock_download):
        mock_fs = MagicMock()
        mock_fs.find.return_value = [
            "bucket/path/a.json",
            "bucket/path/sub/b.parquet",
        ]
        mock_fs.isdir.return_value = False
        mock_fsspec.core.url_to_fs.return_value = (mock_fs, None)

        download_folder("s3://bucket/path", "/local/path")

        assert mock_download.call_count == 2
        mock_download.assert_any_call("s3://bucket/path/a.json", "/local/path/a.json")
        mock_download.assert_any_call(
            "s3://bucket/path/sub/b.parquet", "/local/path/sub/b.parquet"
        )


class TestDownloadSelective:

    @patch("tao_sdk.script_runner.download_file")
    def test_downloads_only_referenced_files(self, mock_download):
        annotations = [
            {"id": "1", "video": "videos/clip_a.mp4"},
            {"id": "2", "video": "videos/clip_b.mp4"},
        ]
        download_selective("s3://bucket/data", "/local/data", annotations, "video")

        assert mock_download.call_count == 2
        mock_download.assert_any_call(
            "s3://bucket/data/videos/clip_a.mp4", "/local/data/videos/clip_a.mp4"
        )
        mock_download.assert_any_call(
            "s3://bucket/data/videos/clip_b.mp4", "/local/data/videos/clip_b.mp4"
        )

    @patch("tao_sdk.script_runner.download_file")
    def test_custom_video_key(self, mock_download):
        annotations = [
            {"filepath": "img_001.png"},
            {"filepath": "img_002.png"},
        ]
        download_selective("s3://bucket/images", "/local/images", annotations, "filepath")

        assert mock_download.call_count == 2

    @patch("tao_sdk.script_runner.download_file")
    def test_skips_empty_paths(self, mock_download):
        annotations = [
            {"video": "clip.mp4"},
            {"video": ""},
            {"other_key": "val"},
        ]
        download_selective("s3://bucket/data", "/local/data", annotations, "video")

        assert mock_download.call_count == 1


class TestUploadResults:

    @patch("tao_sdk.script_runner.fsspec")
    @patch("builtins.open", mock_open(read_data=b"data"))
    def test_uploads_all_files(self, mock_fsspec, tmp_path):
        # Create local files
        results = tmp_path / "results"
        results.mkdir()
        (results / "output.json").write_text("{}")
        (results / "sub").mkdir()
        (results / "sub" / "model.pt").write_bytes(b"model")

        mock_dst = MagicMock()
        mock_fsspec.open.return_value.__enter__ = MagicMock(return_value=mock_dst)
        mock_fsspec.open.return_value.__exit__ = MagicMock(return_value=False)

        upload_results(str(results), "s3://bucket/results/job-1")

        assert mock_fsspec.open.call_count == 2

    @patch("tao_sdk.script_runner.fsspec")
    @patch("builtins.open", mock_open(read_data=b"data"))
    def test_excludes_patterns(self, mock_fsspec, tmp_path):
        results = tmp_path / "results"
        results.mkdir()
        (results / "output.json").write_text("{}")
        (results / "temp.tmp").write_text("tmp")
        (results / "big.tar.gz").write_bytes(b"tar")

        mock_dst = MagicMock()
        mock_fsspec.open.return_value.__enter__ = MagicMock(return_value=mock_dst)
        mock_fsspec.open.return_value.__exit__ = MagicMock(return_value=False)

        upload_results(str(results), "s3://bucket/out", excludes=["*.tmp", "*.tar.gz"])

        # Only output.json should be uploaded
        assert mock_fsspec.open.call_count == 1


class TestArchiveExtraction:

    def _make_tar(self, path, payload_dirname="videos"):
        """Create a tar.gz with a top-level `<payload_dirname>/` dir and one file."""
        import tarfile
        import io
        tar_bytes = io.BytesIO()
        with tarfile.open(fileobj=tar_bytes, mode="w:gz") as tar:
            data = b"hello"
            info = tarfile.TarInfo(name=f"{payload_dirname}/a.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        path.write_bytes(tar_bytes.getvalue())

    def _make_root_tar(self, path):
        """Create a tar.gz with one file at archive root."""
        import tarfile
        import io
        tar_bytes = io.BytesIO()
        with tarfile.open(fileobj=tar_bytes, mode="w:gz") as tar:
            data = b"hello"
            info = tarfile.TarInfo(name="a.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        path.write_bytes(tar_bytes.getvalue())

    def test_is_archive(self):
        assert _is_archive("x.tar.gz")
        assert _is_archive("x.tar")
        assert _is_archive("x.tgz")
        assert not _is_archive("x.mp4")
        assert not _is_archive("folder/")

    def test_strip_archive_suffix(self):
        assert _strip_archive_suffix("/a/b/videos.tar.gz") == "/a/b/videos"
        assert _strip_archive_suffix("/a/videos.tar") == "/a/videos"
        assert _strip_archive_suffix("/a/videos.tgz") == "/a/videos"
        assert _strip_archive_suffix("/a/videos.mp4") == "/a/videos.mp4"

    def test_extract_archive_creates_contents_and_marker(self, tmp_path):
        tar_path = tmp_path / "videos.tar.gz"
        self._make_tar(tar_path)

        extract_archive(str(tar_path), str(tmp_path))

        assert (tmp_path / "videos" / "a.txt").read_bytes() == b"hello"
        assert (tmp_path / "videos.tar.gz.extracted").exists()

    def test_extract_archive_root_files_land_in_archive_stem(self, tmp_path):
        tar_path = tmp_path / "images.tar.gz"
        self._make_root_tar(tar_path)

        extract_archive(str(tar_path), str(tmp_path))

        assert (tmp_path / "images" / "a.txt").read_bytes() == b"hello"
        assert (tmp_path / "images.tar.gz.extracted").exists()

    def test_extract_archive_is_idempotent(self, tmp_path):
        tar_path = tmp_path / "videos.tar.gz"
        self._make_tar(tar_path)
        extract_archive(str(tar_path), str(tmp_path))

        # Remove extracted content; stale marker should trigger re-extract.
        import shutil
        shutil.rmtree(tmp_path / "videos")
        extract_archive(str(tar_path), str(tmp_path))
        assert (tmp_path / "videos" / "a.txt").read_bytes() == b"hello"

    def test_extract_archives_in_dir_walks(self, tmp_path):
        sub = tmp_path / "data"
        sub.mkdir()
        self._make_tar(sub / "videos.tar.gz", payload_dirname="videos")
        (sub / "readme.txt").write_text("no archive here")

        extract_archives_in_dir(str(sub))
        assert (sub / "videos" / "a.txt").exists()


class TestRewriteSpecPaths:

    def test_rewrite_paths(self):
        """Verify _set_nested rewrites S3 paths correctly."""
        specs = {
            "dataset": {
                "annotation_path": "s3://bucket/ann.json",
            },
            "results_dir": "s3://bucket/results/job-1",
        }
        # Simulate what run() does
        _set_nested(specs, "dataset.annotation_path", "/results/test/inputs/ann.json")
        _set_nested(specs, "results_dir", "/results/test/results_dir")

        assert specs["dataset"]["annotation_path"] == "/results/test/inputs/ann.json"
        assert specs["results_dir"] == "/results/test/results_dir"
        assert not _is_remote(specs["dataset"]["annotation_path"])
        assert not _is_remote(specs["results_dir"])


class TestSelectiveWithAnnotation:

    def test_selective_logic(self):
        """Verify selective download uses annotation data and video_key."""
        annotation_data = [
            {"id": "1", "video": "clip_a.mp4"},
            {"id": "2", "video": "clip_b.mp4"},
        ]
        with patch("tao_sdk.script_runner.download_file") as mock_dl:
            download_selective(
                "s3://bucket/videos", "/local/videos",
                annotation_data, video_key="video",
            )
        assert mock_dl.call_count == 2
        mock_dl.assert_any_call(
            "s3://bucket/videos/clip_a.mp4", "/local/videos/clip_a.mp4"
        )
        mock_dl.assert_any_call(
            "s3://bucket/videos/clip_b.mp4", "/local/videos/clip_b.mp4"
        )


def _mock_fsspec_url_to_fs(file_paths=None):
    """Create a mock fsspec filesystem that knows which paths are files vs dirs."""
    file_paths = file_paths or set()
    mock_fs = MagicMock()
    mock_fs.isdir.side_effect = lambda path: path not in file_paths and not any(
        path.endswith(ext) for ext in ('.json', '.csv', '.pth', '.parquet', '.txt', '.yaml', '.toml')
    )
    return mock_fs, ""


class TestMountMode:

    @patch("os.makedirs")
    @patch("tao_sdk.script_runner.subprocess")
    @patch("tao_sdk.script_runner.download_file")
    @patch("builtins.open", mock_open(read_data="{}"))
    def test_skip_download_when_lustre_exists(self, mock_dl, mock_sub, mock_makedirs):
        """When data exists on mount path, skip download."""
        mock_sub.run.return_value = MagicMock(returncode=0)

        specs = {
            "dataset": {
                "annotation_path": "s3://bucket/data/ann.json",
            },
        }
        inputs = ["dataset.annotation_path"]
        outputs = []

        with patch("os.path.exists", return_value=True):
            run(
                specs=specs, job_id="test-mount",
                command="echo {config_path}",
                inputs=inputs, outputs=outputs,
                config_format="json",
                mount_path="/scratch/fsw",
            )

        mock_dl.assert_not_called()
        assert specs["dataset"]["annotation_path"] == "/scratch/fsw/bucket/data/ann.json"

    @patch("os.makedirs")
    @patch("tao_sdk.script_runner.subprocess")
    @patch("tao_sdk.script_runner.download_file")
    @patch("builtins.open", mock_open(read_data="{}"))
    def test_lustre_miss_downloads_to_lustre(self, mock_dl, mock_sub, mock_makedirs):
        """When data is NOT on Lustre, download from S3 to Lustre path."""
        mock_sub.run.return_value = MagicMock(returncode=0)

        specs = {
            "dataset": {
                "annotation_path": "s3://mybucket/eval/ann.json",
            },
        }
        inputs = ["dataset.annotation_path"]
        outputs = {}

        with patch("os.path.exists", return_value=False):
            run(
                specs=specs, job_id="test-miss",
                command="echo {config_path}",
                inputs=inputs, outputs=outputs,
                config_format="json",
                mount_path="/mnt/lustre",
            )

        # Should download to Lustre path (with bucket in path)
        mock_dl.assert_called_once_with(
            "s3://mybucket/eval/ann.json",
            "/mnt/lustre/mybucket/eval/ann.json",
        )
        assert specs["dataset"]["annotation_path"] == "/mnt/lustre/mybucket/eval/ann.json"

    @patch("os.makedirs")
    @patch("tao_sdk.script_runner.subprocess")
    @patch("tao_sdk.script_runner.download_selective")
    @patch("tao_sdk.script_runner.download_file")
    def test_lustre_selective_download(self, mock_dl_file, mock_dl_sel, mock_sub, mock_makedirs):
        """Selective folder download goes to Lustre when mount exists."""
        mock_sub.run.return_value = MagicMock(returncode=0)

        ann_data = json.dumps([
            {"id": "1", "video": "videos/a.mp4"},
            {"id": "2", "video": "videos/b.mp4"},
        ])

        specs = {
            "dataset": {
                "annotation_path": "s3://bucket/eval/ann.json",
                "media_dir": "s3://bucket/eval/",
            },
        }
        inputs = {
            "dataset.annotation_path": {"type": "file"},
            "dataset.media_dir": {
                "type": "folder",
                "selective": {
                    "annotation": "dataset.annotation_path",
                    "format": "json",
                    "keys": ["video"],
                },
            },
        }
        outputs = {}

        def fake_exists(path):
            # Annotation exists on Lustre, videos do NOT
            if path.endswith("ann.json"):
                return True
            return False

        with patch("os.path.exists", side_effect=fake_exists), \
             patch("builtins.open", mock_open(read_data=ann_data)):
            run(
                specs=specs, job_id="test-sel",
                command="echo {config_path}",
                inputs=inputs, outputs=outputs,
                config_format="json",
                mount_path="/mnt/lustre",
            )

        # Annotation found on Lustre — no file download
        mock_dl_file.assert_not_called()
        # Selective download to Lustre path
        mock_dl_sel.assert_called_once()
        call_args = mock_dl_sel.call_args
        assert call_args[0][0] == "s3://bucket/eval/"  # remote dir
        assert call_args[0][1] == "/mnt/lustre/bucket/eval/"  # lustre dir

    @patch("os.makedirs")
    @patch("tao_sdk.script_runner.subprocess")
    @patch("tao_sdk.script_runner.download_selective")
    @patch("tao_sdk.script_runner.download_file")
    def test_lustre_selective_all_present_skips_download(self, mock_dl_file, mock_dl_sel, mock_sub, mock_makedirs):
        """When all selective files exist on Lustre, skip download entirely."""
        mock_sub.run.return_value = MagicMock(returncode=0)

        ann_data = json.dumps([
            {"id": "1", "video": "videos/a.mp4"},
            {"id": "2", "video": "videos/b.mp4"},
        ])

        specs = {
            "dataset": {
                "annotation_path": "s3://bucket/eval/ann.json",
                "media_dir": "s3://bucket/eval/",
            },
        }
        inputs = {
            "dataset.annotation_path": {"type": "file"},
            "dataset.media_dir": {
                "type": "folder",
                "selective": {
                    "annotation": "dataset.annotation_path",
                    "format": "json",
                    "keys": ["video"],
                },
            },
        }
        outputs = {}

        # Everything exists on Lustre
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data=ann_data)):
            run(
                specs=specs, job_id="test-all-present",
                command="echo {config_path}",
                inputs=inputs, outputs=outputs,
                config_format="json",
                mount_path="/mnt/lustre",
            )

        mock_dl_file.assert_not_called()
        mock_dl_sel.assert_not_called()
        assert specs["dataset"]["media_dir"] == "/mnt/lustre/bucket/eval/"

    @patch("os.makedirs")
    @patch("tao_sdk.script_runner.subprocess")
    @patch("tao_sdk.script_runner.download_folder")
    @patch("tao_sdk.script_runner.download_file")
    def test_lustre_folder_miss_downloads_entire_folder(self, mock_dl_file, mock_dl_folder, mock_sub, mock_makedirs):
        """Non-selective folder miss downloads entire folder to Lustre."""
        mock_sub.run.return_value = MagicMock(returncode=0)

        specs = {
            "model": {
                "weights_dir": "s3://bucket/models/v1/",
            },
        }
        inputs = ["model.weights_dir"]
        outputs = {}

        def fake_exists(path):
            return False

        with patch("os.path.exists", side_effect=fake_exists), \
             patch("os.listdir", return_value=[]), \
             patch("builtins.open", mock_open(read_data="{}")):
            run(
                specs=specs, job_id="test-folder-miss",
                command="echo {config_path}",
                inputs=inputs, outputs=outputs,
                config_format="json",
                mount_path="/mnt/lustre",
            )

        mock_dl_folder.assert_called_once_with(
            "s3://bucket/models/v1/",
            "/mnt/lustre/bucket/models/v1/",
        )
        assert specs["model"]["weights_dir"] == "/mnt/lustre/bucket/models/v1/"

    @patch("os.makedirs")
    @patch("tao_sdk.script_runner.subprocess")
    @patch("tao_sdk.script_runner.download_file")
    @patch("builtins.open", mock_open(read_data="{}"))
    def test_no_mount_downloads_to_container_local(self, mock_dl, mock_sub, mock_makedirs):
        """Without mount_path, downloads go to container-local /results."""
        mock_sub.run.return_value = MagicMock(returncode=0)

        specs = {
            "dataset": {
                "annotation_path": "s3://bucket/data/ann.json",
            },
        }
        inputs = ["dataset.annotation_path"]
        outputs = {}

        with patch("os.path.exists", return_value=False):
            run(
                specs=specs, job_id="test-local",
                command="echo {config_path}",
                inputs=inputs, outputs=outputs,
                config_format="json",
                mount_path=None,
            )

        # Should download to /results/test-local/inputs/...
        mock_dl.assert_called_once()
        dest = mock_dl.call_args[0][1]
        assert dest.startswith("/results/test-local/inputs/")
        assert "lustre" not in dest


class TestArchiveInRun:
    """End-to-end: run() downloads a .tar.gz file input, extracts it, and
    rewrites the spec to the extracted directory (TAO Core convention)."""

    def _make_tar(self, path, payload_dirname="videos"):
        import tarfile
        import io
        tar_bytes = io.BytesIO()
        with tarfile.open(fileobj=tar_bytes, mode="w:gz") as tar:
            data = b"hello"
            info = tarfile.TarInfo(name=f"{payload_dirname}/a.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        path.write_bytes(tar_bytes.getvalue())

    def test_file_tar_gz_extracted_and_spec_rewritten(self, tmp_path):
        """File input ending in .tar.gz: extract into parent, rewrite spec to
        the path with .tar.gz stripped."""
        from pathlib import Path

        # download_file: write a real tar.gz to the local destination
        def fake_download_file(remote, local):
            os.makedirs(os.path.dirname(local), exist_ok=True)
            self._make_tar(Path(local))

        def fake_url_to_fs(url):
            fs = MagicMock()
            fs.isdir.return_value = False
            return fs, ""

        # os.makedirs should create real dirs under tmp_path; skip /results/
        real_makedirs = os.makedirs

        def safe_makedirs(path, *args, **kwargs):
            if str(path).startswith("/results"):
                return
            return real_makedirs(path, *args, **kwargs)

        specs = {"custom": {"train_dataset": {"media_path": "s3://bucket/data/videos.tar.gz"}}}
        inputs = {"custom.train_dataset.media_path": {"type": "file"}}

        real_open = open

        def safe_open(path, *args, **kwargs):
            if isinstance(path, str) and path.startswith("/results"):
                return mock_open()(path, *args, **kwargs)
            return real_open(path, *args, **kwargs)

        with patch("tao_sdk.script_runner.download_file", side_effect=fake_download_file), \
             patch("tao_sdk.script_runner.fsspec.core.url_to_fs", side_effect=fake_url_to_fs), \
             patch("tao_sdk.script_runner.subprocess") as mock_sub, \
             patch("tao_sdk.script_runner.os.makedirs", side_effect=safe_makedirs), \
             patch("builtins.open", side_effect=safe_open):
            mock_sub.run.return_value = MagicMock(returncode=0)
            run(
                specs=specs, job_id="test-archive",
                command="true",
                inputs=inputs, outputs={},
                config_format="json",
                mount_path=str(tmp_path / "lustre"),
            )

        # Spec rewritten to the extracted directory (TAO Core convention)
        expected = str(tmp_path / "lustre" / "bucket" / "data" / "videos")
        assert specs["custom"]["train_dataset"]["media_path"] == expected
        assert (tmp_path / "lustre" / "bucket" / "data" / "videos" / "a.txt").read_bytes() == b"hello"


class TestOutputDestinationBranches:
    """The fix for the cosmos-rl silent-skip bug: run() now has three
    explicit branches for output spec values — remote URI (upload),
    absolute local path (write directly, no upload), and unresolved
    (raise loudly). Verifies the branch selection without running a
    real command. ``os.makedirs`` is mocked because script_runner
    hardcodes ``/results/<job_id>`` which is not writable on the
    test host (read-only on macOS)."""

    def _common_specs(self, output_dir_value):
        return {"train": {"output_dir": output_dir_value, "epochs": 1}}

    @patch("tao_sdk.script_runner.os.path.isdir", return_value=True)
    @patch("tao_sdk.script_runner.os.makedirs")
    @patch("tao_sdk.script_runner.subprocess.run")
    @patch("tao_sdk.script_runner.upload_results")
    @patch("tao_sdk.script_runner.open", create=True, new_callable=mock_open)
    def test_remote_uri_rewrites_to_local_and_queues_upload(
        self, _mock_open, mock_upload, mock_subprocess, _mock_mkdir, _mock_isdir,
    ):
        mock_subprocess.return_value = MagicMock(returncode=0)
        specs = self._common_specs("s3://my-bucket/results/job-1/output")
        run(
            specs=specs, job_id="job-1",
            command="echo {config_path}",
            inputs={}, outputs={"train.output_dir": {"type": "folder"}},
            config_format="json",  # stdlib, no toml import inside run()
        )
        # Spec rewritten to a local path under /results/<job>/...
        assert specs["train"]["output_dir"].startswith("/results/job-1/")
        # Upload was queued back to the original remote URI.
        mock_upload.assert_called_once()
        assert mock_upload.call_args[0][1] == "s3://my-bucket/results/job-1/output"

    @patch("tao_sdk.script_runner.os.makedirs")
    @patch("tao_sdk.script_runner.subprocess.run")
    @patch("tao_sdk.script_runner.upload_results")
    @patch("tao_sdk.script_runner.open", create=True, new_callable=mock_open)
    def test_absolute_local_path_used_directly_no_upload(
        self, _mock_open, mock_upload, mock_subprocess, mock_mkdir,
    ):
        """Lustre / K8s PVC / Docker bind mount case — script_runner uses
        the absolute path directly and skips upload entirely."""
        mock_subprocess.return_value = MagicMock(returncode=0)
        lustre_path = "/lustre/fsw/tao/results/job-1/train_output_dir"
        specs = self._common_specs(lustre_path)
        run(
            specs=specs, job_id="job-1",
            command="echo {config_path}",
            inputs={}, outputs={"train.output_dir": {"type": "folder"}},
            config_format="json",
        )
        # Spec preserved as the absolute Lustre path.
        assert specs["train"]["output_dir"] == lustre_path
        # No upload — the platform's mount handles persistence.
        mock_upload.assert_not_called()
        # script_runner created the lustre dir so the model can write there.
        mkdir_calls = [c.args[0] for c in mock_mkdir.call_args_list]
        assert lustre_path in mkdir_calls

    @patch("tao_sdk.script_runner.os.makedirs")
    @patch("tao_sdk.script_runner.subprocess.run")
    @patch("tao_sdk.script_runner.upload_results")
    @patch("tao_sdk.script_runner.open", create=True, new_callable=mock_open)
    def test_relative_default_autofills_to_s3_when_bucket_set(
        self, _mock_open, mock_upload, mock_subprocess, _mock_mkdir, monkeypatch,
    ):
        """The cosmos-rl bug case under runtime resolution: relative ``output_dir``
        gets auto-filled to ``s3://$S3_BUCKET_NAME/results/<job_id>/...`` and
        upload happens at end of run."""
        monkeypatch.setenv("S3_BUCKET_NAME", "my-bucket")
        monkeypatch.delenv("TAO_RESULTS_ROOT", raising=False)
        mock_subprocess.return_value = MagicMock(returncode=0)
        specs = self._common_specs("output")  # the cosmos-rl bug shape
        with patch("tao_sdk.script_runner.os.path.isdir", return_value=True):
            run(
                specs=specs, job_id="job-1",
                command="echo {config_path}",
                inputs={}, outputs={"train.output_dir": {"type": "folder"}},
                config_format="json",
            )
        assert specs["train"]["output_dir"].startswith("/results/job-1/")
        mock_upload.assert_called_once()
        assert mock_upload.call_args[0][1] == "s3://my-bucket/results/job-1/train_output_dir"

    @patch("tao_sdk.script_runner.os.makedirs")
    @patch("tao_sdk.script_runner.subprocess.run")
    @patch("tao_sdk.script_runner.upload_results")
    @patch("tao_sdk.script_runner.open", create=True, new_callable=mock_open)
    def test_relative_default_autofills_to_mount_when_results_root_set(
        self, _mock_open, mock_upload, mock_subprocess, _mock_mkdir, monkeypatch,
    ):
        """SLURM/Lustre / K8s PVC / Docker bind: TAO_RESULTS_ROOT is set, so
        the auto-fill targets the mount and no upload happens."""
        monkeypatch.setenv("TAO_RESULTS_ROOT", "/lustre/fsw/tao/results")
        mock_subprocess.return_value = MagicMock(returncode=0)
        specs = self._common_specs("output")
        run(
            specs=specs, job_id="job-1",
            command="echo {config_path}",
            inputs={}, outputs={"train.output_dir": {"type": "folder"}},
            config_format="json",
        )
        assert specs["train"]["output_dir"] == "/lustre/fsw/tao/results/job-1/train_output_dir"
        mock_upload.assert_not_called()

    @patch("tao_sdk.script_runner.os.makedirs")
    @patch("tao_sdk.script_runner.subprocess.run")
    @patch("tao_sdk.script_runner.upload_results")
    @patch("tao_sdk.script_runner.open", create=True, new_callable=mock_open)
    def test_relative_default_falls_back_to_ephemeral_when_nothing_set(
        self, _mock_open, mock_upload, mock_subprocess, _mock_mkdir, monkeypatch, caplog,
    ):
        """No TAO_RESULTS_ROOT, no S3_BUCKET_NAME → container-local
        ``/results/<job_id>/`` with a loud warning. Job proceeds (smoke / dev
        runs work without env setup) but the warning makes the data loss
        visible — replaces the original silent zero-upload bug."""
        import logging
        monkeypatch.delenv("TAO_RESULTS_ROOT", raising=False)
        monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
        mock_subprocess.return_value = MagicMock(returncode=0)
        specs = self._common_specs("output")
        with caplog.at_level(logging.WARNING, logger="tao_sdk.script_runner"):
            run(
                specs=specs, job_id="job-1",
                command="echo {config_path}",
                inputs={}, outputs={"train.output_dir": {"type": "folder"}},
                config_format="json",
            )
        assert specs["train"]["output_dir"] == "/results/job-1/train_output_dir"
        mock_upload.assert_not_called()
        # Loud end-of-run warning so silent-loss-via-ephemeral is impossible.
        assert any("NOT survive" in r.message for r in caplog.records)


class TestS3ToLustreMapping:

    def test_basic_mapping(self):
        assert _s3_to_lustre("s3://bucket/path/file.json", "/mnt/lustre") == \
            "/mnt/lustre/bucket/path/file.json"

    def test_nested_path(self):
        assert _s3_to_lustre("s3://nvcf-storage/data/sda/collision_eval/ann.json", "/mnt/lustre") == \
            "/mnt/lustre/nvcf-storage/data/sda/collision_eval/ann.json"

    def test_different_mount(self):
        assert _s3_to_lustre("s3://bucket/file.txt", "/scratch/fsw") == \
            "/scratch/fsw/bucket/file.txt"

    def test_trailing_slash(self):
        result = _s3_to_lustre("s3://bucket/folder/", "/mnt/lustre")
        assert result == "/mnt/lustre/bucket/folder/"
