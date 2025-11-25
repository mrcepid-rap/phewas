"""
This runs the PheWAS association tests using the LoadModule class from phewas.
"""

import shutil
from pathlib import Path

import pytest

from phewas.loader import LoadModule

KEEP_TEMP = True

test_data_dir = Path(__file__).parent / 'test_data'


@pytest.fixture
def temporary_path(tmp_path, monkeypatch):
    """
    Prepare a temporary working directory that contains a copy of the test_data
    directory, then change the working directory to it.

    If KEEP_TEMP is True, after the test the entire temporary directory will be copied
    to a folder 'temp_test_outputs' in the project root.
    """
    test_data_source = Path(__file__).parent / "test_data"

    destination = tmp_path / "test_data"
    destination.parent.mkdir(parents=True, exist_ok=True)

    shutil.copytree(test_data_source, destination)

    monkeypatch.chdir(tmp_path)

    yield tmp_path

    if KEEP_TEMP:
        persistent_dir = Path(__file__).parent / "temp_test_outputs" / tmp_path.name
        persistent_dir.parent.mkdir(exist_ok=True)
        shutil.copytree(tmp_path, persistent_dir, dirs_exist_ok=True)
        print(f"Temporary output files have been copied to: {persistent_dir}")


import os

@pytest.mark.parametrize("input_args", [
    (
            f"--association_tarballs {test_data_dir}/HC_PTV-MAF_001.tar.gz "
            f"--bgen_index {test_data_dir}/bgen_locs.tsv "
            f"--sparse_grm {test_data_dir}/sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx "
            f"--sparse_grm_sample {test_data_dir}/sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx.sampleIDs.txt "
            f"--gene_ids OR4F5 "
            f"--phenofile {test_data_dir}/phenotype.tsv "
            f"--transcript_index {test_data_dir}/transcripts.tsv.gz "
            f"--base_covariates {test_data_dir}/base_covariates.covariates "
    ),
])
def test_phewas_load_module_run(input_args, temporary_path):
    """
    Test the PheWAS LoadModule: initialization, running start_module, and outputs.
    Also check that expected output files exist.
    """
    loader = LoadModule(output_prefix="test", input_args=input_args)
    loader.start_module()
    outputs = loader.get_outputs()
    assert outputs is not None, "Outputs should not be None"
    print(outputs)

    expected_dir = test_data_dir.parent / "expected_output"
    expected_files = [
        "phenotypes_covariates.formatted.txt",
        "test.genes.STAAR.stats.tsv.gz",
        "test.genes.STAAR.stats.tsv.gz.tbi",
        "test.genes.STAAR_glm.stats.tsv.gz",
        "test.genes.STAAR_glm.stats.tsv.gz.tbi",
    ]
    for fname in expected_files:
        file_path = expected_dir / fname
        assert file_path.exists(), f"Expected file missing: {file_path}"
