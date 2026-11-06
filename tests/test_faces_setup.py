"""Verify complete prerequisite reporting and extraction/download placement without network or large assets."""

from __future__ import annotations

import contextlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mfdime import setup_faces_pilot as setup


class FaceSetupTests(unittest.TestCase):
    def test_audit_lists_all_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = [root / "pmf.pt", root / "steex.tar", root / "faces.zip"]
            with patch.object(setup, "required_files", return_value=missing), \
                 patch.object(setup.clip_smile, "DIRECTORY", root / "clip"), \
                 patch.object(setup.config, "FIRE_REPO", root / "fire"), \
                 patch.object(setup.config, "CELEBAHQ_ZIP", root / "faces.zip"):
                issues = setup.audit()
            for path in missing:
                self.assertTrue(any(str(path) in message for message in issues))

    def test_check_only_performs_no_setup_tasks(self):
        output = io.StringIO()
        with patch.object(setup, "audit", return_value=["missing pMF", "missing STEEX", "missing dataset"]), \
             patch.object(setup, "setup_fire", side_effect=AssertionError("network/setup called")), \
             contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
            setup.main(["--check-only"])
        self.assertIn("missing pMF", output.getvalue())
        self.assertIn("missing STEEX", output.getvalue())
        self.assertIn("missing dataset", output.getvalue())

    def test_only_requested_steex_member_is_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "release.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                for name, content in [("checkpoints/" + setup.STEEX_MEMBER, b"smile checkpoint"),
                                      ("../../unexpected.txt", b"do not extract"),
                                      ("decision_densenet/celeba/checkpoint.tar", b"other dataset")]:
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    stream.addfile(info, io.BytesIO(content))
            dest = root / "weights/steex/decision_densenet/celebamaskhq/checkpoint.tar"
            setup.extract_steex(archive, dest)
            self.assertEqual(dest.read_bytes(), b"smile checkpoint")
            self.assertEqual(len(list((root / "weights").rglob("*checkpoint.tar"))), 1)
            self.assertFalse((root / "unexpected.txt").exists())

    def test_existing_download_is_reused_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "pmf.pt"
            dest.write_bytes(b"existing weights")
            with patch.object(setup, "hf_hub_download", side_effect=AssertionError("downloaded again")):
                setup.download_hf("repo", "revision", "model", "pmf.pt", dest)

    def test_setup_covers_all_assets_and_reports_multiple_failures(self):
        called = []

        def hf(repo, revision, kind, filename, destination):
            called.append((repo, kind, filename))

        output = io.StringIO()
        with patch.object(setup, "setup_fire"), patch.object(setup, "download_hf", side_effect=hf), \
             patch.object(setup, "setup_steex", side_effect=RuntimeError("STEEX unavailable")), \
             patch.object(setup, "setup_vgg", side_effect=RuntimeError("VGG unavailable")), \
             patch.object(setup, "setup_clip") as clip, patch.object(setup, "audit", return_value=["missing STEEX", "missing VGG"]), \
             contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
            setup.main([])
        self.assertEqual(len(called), 3)  # pMF, HQ zip, and partition
        self.assertEqual([kind for _, kind, _ in called], ["model", "dataset", "dataset"])
        self.assertIn("STEEX unavailable", output.getvalue())
        self.assertIn("VGG unavailable", output.getvalue())
        clip.assert_called_once()


if __name__ == "__main__":
    unittest.main()
