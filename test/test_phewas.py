"""
This runs the PheWAS association tests using the LoadModule class from phewas.
"""

import shutil
from pathlib import Path

import pandas as pd
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


@pytest.mark.parametrize("output_prefix, input_args, expected_files", [
    # the first tests a single gene with one phenotype
    (
            "test",
            (
                    "--association_tarballs test_data/HC_PTV-MAF_001.tar.gz "
                    "--bgen_index test_data/bgen_locs.tsv "
                    "--sparse_grm test_data/sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx "
                    "--sparse_grm_sample test_data/sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx.sampleIDs.txt "
                    "--gene_ids OR4F5 "
                    "--phenofile test_data/phenotype.tsv "
                    "--transcript_index test_data/transcripts.tsv.gz "
                    "--base_covariates test_data/base_covariates.covariates"
            ),
            ["test.genes.glm.stats.tsv.gz", "test.genes.STAAR.stats.tsv.gz", "test.genes.STAAR_glm.stats.tsv.gz"]
    ),
    # the second tests a single gene with two phenotypes
    (
            "test2",
            (
                    "--association_tarballs test_data/HC_PTV-MAF_001.tar.gz "
                    "--bgen_index test_data/bgen_locs.tsv "
                    "--sparse_grm test_data/sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx "
                    "--sparse_grm_sample test_data/sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx.sampleIDs.txt "
                    "--gene_ids OR4F5 "
                    "--phenofile test_data/phenotype2.tsv "
                    "--transcript_index test_data/transcripts.tsv.gz "
                    "--base_covariates test_data/base_covariates.covariates"
            ),
            ["test2.genes.glm.stats.tsv.gz", "test2.genes.STAAR.stats.tsv.gz", "test2.genes.STAAR_glm.stats.tsv.gz"]
    ),
    # the third tests multiple genes with two phenotypes
    (
            "test3",
            (
                    "--association_tarballs test_data/HC_PTV-MAF_001.tar.gz "
                    "--bgen_index test_data/bgen_locs.tsv "
                    "--sparse_grm test_data/sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx "
                    "--sparse_grm_sample test_data/sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx.sampleIDs.txt "
                    "--gene_ids OR4F5 LAPTM5 "
                    "--phenofile test_data/phenotype2.tsv "
                    "--transcript_index test_data/transcripts.tsv.gz "
                    "--base_covariates test_data/base_covariates.covariates"
            ),
            ["test3.genes.glm.stats.tsv.gz", "test3.genes.STAAR.stats.tsv.gz", "test3.genes.STAAR_glm.stats.tsv.gz"]
    ),
])
def test_phewas_load_module_run(output_prefix, input_args, expected_files, temporary_path):
    # 1. Run the module
    loader = LoadModule(output_prefix=output_prefix, input_args=input_args)
    loader.start_module()

    outputs = loader.get_outputs()
    assert outputs is not None

    # Reference the expected_output folder relative to the test file
    expected_dir = Path(__file__).parent / "expected_output"

    # 2. Iterate through files and compare
    for fname in expected_files:
        expected_file = expected_dir / fname
        actual_file = temporary_path / fname

        assert expected_file.exists(), f"Expected file {expected_file} does not exist"
        assert actual_file.exists(), f"Generated file {actual_file} does not exist"

        df_expected = pd.read_csv(expected_file, sep='\t')
        df_actual = pd.read_csv(actual_file, sep='\t')

        # Sort by all columns to ensure that the order is the same
        df_expected = df_expected.sort_values(by=df_expected.columns.to_list()).reset_index(drop=True)
        df_actual = df_actual.sort_values(by=df_actual.columns.to_list()).reset_index(drop=True)
        pd.testing.assert_frame_equal(df_expected, df_actual)

    # 3. Check for .tbi index files
    for fname in expected_files:
        tbi_file = temporary_path / f"{fname}.tbi"
        assert tbi_file.exists(), f"Index file {tbi_file.name} missing"
