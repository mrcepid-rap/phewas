from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import dxpy
import pandas as pd
from general_utilities.association_resources import (
    bgzip_and_tabix,
    build_transcript_table,
    get_gene_id,
    process_gene_or_snp_wgs,
    process_snp_or_gene_tar,
)
from general_utilities.bgen_utilities.genotype_matrix import generate_csr_matrix_from_bgen
from general_utilities.import_utils.file_handlers.export_file_handler import ExportFileHandler
from general_utilities.import_utils.file_handlers.input_file_handler import InputFileHandler
from general_utilities.import_utils.import_lib import TarballType
from general_utilities.job_management.joblauncher_factory import joblauncher_factory
from general_utilities.job_management.thread_utility import ThreadUtility
from general_utilities.linear_model import linear_model
from general_utilities.linear_model.linear_model import LinearModelResult
from general_utilities.linear_model.proccess_model_output import merge_glm_staar_runs, process_model_outputs
from general_utilities.linear_model.staar_model import load_staar_genetic_data, staar_genes, staar_null
from general_utilities.mrc_logger import MRCLogger
from scipy.io import mmwrite

from phewas.phewas_association_pack import PhewasAssociationPack


class PheWAS:

    def __init__(self, output_prefix: str, association_pack: PhewasAssociationPack):

        self._logger = MRCLogger(__name__).get_logger()
        self._association_pack = association_pack
        self._output_prefix = output_prefix
        self._outputs = []

        # Initialize data structures
        self.genetic_map = defaultdict(list)

        # build the transcripts table
        self._transcripts_table = build_transcript_table(transcripts_path=self._association_pack.transcript_index)

        # Figure out genes/SNPlist to run...
        if self._association_pack.tarball_type in (TarballType.SNP, TarballType.GENE):
            self._logger.info("Initializing non-standard tarball extraction (SNP/GENE tar)")
            gene_info, returned_chromosomes = process_snp_or_gene_tar(
                self._association_pack.tarball_type == TarballType.SNP,
                self._association_pack.tarball_type == TarballType.GENE,
                self._association_pack.tarball_prefixes[0]
            )
            for chromosome in returned_chromosomes:
                self.genetic_map[chromosome].append(gene_info)
        else:
            for gene_id in self._association_pack.gene_ids:
                # get_gene_id handles gene symbols and ENST IDs
                gene_info = get_gene_id(gene_id, self._transcripts_table)

                # Search for this gene across all chunks
                for chunk in self._association_pack.bgen_dict:
                    try:
                        chromosomes = process_gene_or_snp_wgs(
                            identifier=gene_info.name,
                            tarball_prefix=self._association_pack.tarball_prefixes[0],
                            chunk=chunk
                        )
                    except FileNotFoundError:
                        self._logger.debug(f"Variant table for chunk {chunk} not found, skipping.")
                        continue

                    if chromosomes:
                        self._logger.info(
                            f"{gene_info['SYMBOL']} found in {chunk} "
                            f"({', '.join(chromosomes)})"
                        )
                        self.genetic_map[chunk].append(gene_info)

    def _add_output(self, file: Path) -> None:
        self._outputs.append(file)

    def _extend_output(self, files: List[Path]) -> None:
        self._outputs.extend(files)

    def get_outputs(self) -> List[Path]:
        return self._outputs

    def run_tool(self):

        self._add_output(Path('phenotypes_covariates.formatted.txt'))

        self._logger.info("Submitting linear models...")
        self._run_linear_models()

        self._logger.info("Running STAAR models...")
        self._run_staar_models()

        self._logger.info("Merging GLM/STAAR runs...")
        self._extend_output(
            merge_glm_staar_runs(
                self._output_prefix,
                self._association_pack.tarball_type == TarballType.SNP,
                self._association_pack.tarball_type == TarballType.GENE,
            )
        )

    def _run_linear_models(self) -> None:
        """
        Run linear models for each gene and chromosome combination.
        """
        self._logger.info("Generating null models for all phenotypes...")
        null_models = {}
        for phenoname in self._association_pack.pheno_names:
            null_model = linear_model.linear_model_null(
                phenotype=phenoname,
                phenofile=self._association_pack.final_covariates,
                is_binary=self._association_pack.is_binary,
                found_quantitative_covariates=self._association_pack.found_quantitative_covariates,
                found_categorical_covariates=self._association_pack.found_categorical_covariates
            )
            null_models[phenoname] = null_model

        self._logger.info("Loading linear model genotypes and running models...")
        thread_utility = ThreadUtility(self._association_pack.threads, thread_factor=1)

        for chromosome, genes_in_chunk in self.genetic_map.items():
            # Load data for this chunk
            for tarball_prefix in self._association_pack.tarball_prefixes:
                _, genetic_data = linear_model.load_linear_model_genetic_data(
                    tarball_prefix=tarball_prefix,
                    tarball_type=self._association_pack.tarball_type,
                    bgen_prefix=chromosome,
                )

                for phenoname, null_model in null_models.items():
                    for gene_info in genes_in_chunk:
                        thread_utility.launch_job(
                            function=linear_model.run_linear_model,
                            inputs={
                                'linear_model_pack': null_model,
                                'genotype_table': genetic_data,
                                'gene': gene_info.name,
                                'mask_name': tarball_prefix.name,
                                'is_binary': self._association_pack.is_binary,
                                'always_run_corrected': True
                            },
                            outputs=['gene_dict']
                        )

        thread_utility.submit_and_monitor()

        finished_genes = []
        for result in thread_utility:
            finished_gene: LinearModelResult = result['gene_dict']
            finished_genes.append(finished_gene)

        process_model_outputs(input_models=finished_genes,
                              output_path=Path(f'{self._output_prefix}.genes.glm.stats.tsv'),
                              tarball_type=self._association_pack.tarball_type,
                              transcripts_table=self._transcripts_table)

        self._logger.info(f"Linear model stats written to {self._output_prefix}.genes.glm.stats.tsv")

    def _run_staar_null_models(self, pheno_data: Dict[str, Tuple[pd.DataFrame, set]]) -> None:
        """
        Runs the STAAR Null Models using the specified phenotypes and parameters. This
        method utilizes multi-threading to handle multiple phenotypes in parallel and
        executes the `staar_null` function for each phenotype.

        Phenotype-specific data and parameters are passed to the `staar_null` function.
        The method ensures that all threads are properly submitted and monitored until
        completion.

        :param pheno_data: A dictionary from _prepare_staar_pheno_data
        :raises Exception: If any errors occur while launching jobs or monitoring threads.
        """

        self._logger.info("Running STAAR Null Model(s)...")
        thread_utility = ThreadUtility(self._association_pack.threads, thread_factor=1)
        for phenoname in self._association_pack.pheno_names:
            pheno_merged_df, _ = pheno_data[phenoname]
            pheno_merged_cov_path = Path(f"merged_covariates_for_staar.{phenoname}.tsv")
            pheno_merged_df.to_csv(pheno_merged_cov_path, sep="\t", index=False)
            thread_utility.launch_job(
                function=staar_null,
                inputs={
                    'phenofile': pheno_merged_cov_path,
                    'phenotype': phenoname,
                    'is_binary': self._association_pack.is_binary,
                    'found_quantitative_covariates': self._association_pack.found_quantitative_covariates,
                    'found_categorical_covariates': self._association_pack.found_categorical_covariates,
                    'sex': self._association_pack.sex,
                    'sparse_kinship_file': self._association_pack.sparse_grm,
                    'sparse_kinship_samples': self._association_pack.sparse_grm_sample
                }
            )
        thread_utility.submit_and_monitor()

    def _prepare_staar_pheno_data(self, sample_df: pd.DataFrame, covar_df: pd.DataFrame,
                                  required_cols: List[str]) -> Dict[str, Tuple[pd.DataFrame, set]]:
        """
        Prepares and cleans phenotype-specific data by merging sample and covariate DataFrames,
        removing rows with missing phenotype or required covariate data.

        :param sample_df: DataFrame with sample IDs and related info.
        :param covar_df: DataFrame with covariate data matched by sample IDs.
        :param required_cols: List of required covariate column names.
        :return: A dictionary where keys are phenonames and values are tuples of
                 (the filtered phenotype DataFrame, a set of sample IDs for the null model).
        """

        self._logger.info("Preparing phenotype-specific data files...")
        pheno_data = {}
        for phenoname in self._association_pack.pheno_names:
            # Merge BGEN IDs with Covariates for the current phenotype
            pheno_merged = sample_df.merge(covar_df, how="left", left_on="ID_2", right_on="IID")
            pheno_merged = pheno_merged.drop(columns=["ID_1", "missing"], errors="ignore")
            pheno_merged = pheno_merged.sort_values("ID_2")

            # Remove rows with missing phenotype data for the CURRENT phenotype
            if phenoname in pheno_merged.columns:
                pheno_merged = pheno_merged.dropna(subset=[phenoname])

            # Only check for required covariate columns that actually exist in the dataframe
            current_required_cols = [col for col in required_cols if col in pheno_merged.columns]
            pheno_merged = pheno_merged.dropna(subset=current_required_cols)

            self._logger.info(
                f"For phenotype '{phenoname}', after removing samples with missing data: {len(pheno_merged)} samples remain")

            # Prep data for return
            pheno_merged = pheno_merged.drop(columns=['IID', 'FID'], errors='ignore')
            pheno_merged = pheno_merged.rename(columns={'ID_2': 'FID'})
            pheno_null_model_samples = set(pheno_merged['FID'].astype(str))
            pheno_data[phenoname] = (pheno_merged, pheno_null_model_samples)

        return pheno_data

    def _run_staar_models(self):
        """
        Run STAAR models for each gene and chromosome combination.
        """

        self._logger.info("Creating merged covariates file for STAAR null model...")
        # Grab the first chromosome's sample file just to get the IDs
        first_chrom = next(iter(self.genetic_map))
        sample_path = self._association_pack.bgen_dict[first_chrom]["sample"].get_file_handle()

        # Read sample file (BGEN format usually has 2 header rows, we skip the second type row)
        sample = pd.read_csv(sample_path, sep=r"\s+", header=0, dtype={'ID_2': str})
        sample = sample.drop(columns=["sex"], errors="ignore")
        sample = sample.iloc[1:].reset_index(drop=True)  # Drop the "0 0 0 D D D..." row

        # Read covariates
        covar = pd.read_csv(self._association_pack.final_covariates, sep=' ', header=0, dtype={'IID': str})

        # Determine required columns for filtering
        required_cols = ['age', 'age_squared', 'batch']
        if self._association_pack.sex == 2:
            required_cols.append('sex')
        # Add principal components (PCs)
        pc_cols = [col for col in covar.columns if col.startswith('PC') and col[2:].isdigit()]
        required_cols.extend(pc_cols)
        required_cols.extend(self._association_pack.found_quantitative_covariates)
        required_cols.extend(self._association_pack.found_categorical_covariates)

        # 1. Create all phenotype-specific data files first
        self._logger.info("Preparing phenotype-specific data files...")
        pheno_data = self._prepare_staar_pheno_data(sample, covar, required_cols)

        # 2. Run the STAAR NULL model for each phenotype in parallel
        self._logger.info("Running STAAR Null Model(s)...")
        self._run_staar_null_models(pheno_data)

        # 3. Filter STAAR sample tables for each phenotype and upload them
        self._logger.info("Filtering STAAR sample tables by phenotype...")
        pheno_filtered_samples = {}
        exporter = ExportFileHandler(delete_on_upload=False)
        for phenoname, (_, pheno_null_model_samples) in pheno_data.items():
            pheno_filtered_samples[phenoname] = {}
            for tarball_prefix in self._association_pack.tarball_prefixes:
                pheno_filtered_samples[phenoname][tarball_prefix.name] = {}
                for chromosome in self.genetic_map:
                    sample_path = self._association_pack.bgen_dict[chromosome]["sample"].get_file_handle()
                    bgen_samples = pd.read_csv(sample_path, sep=r'\s+', header=0, dtype={'ID_2': str})
                    bgen_samples = bgen_samples.iloc[1:].reset_index(drop=True)
                    bgen_samples = bgen_samples.rename(columns={'ID_2': 'FID'})

                    base_samples_path = Path(f"{tarball_prefix}.{chromosome}.STAAR.samples_table.tsv")
                    if not base_samples_path.exists():
                        self._logger.warning(
                            f"Could not find STAAR samples table: {base_samples_path}, skipping filtering.")
                        continue

                    staar_samples_df = pd.read_csv(base_samples_path, sep='\t')
                    staar_samples_df = staar_samples_df.merge(bgen_samples['FID'], left_on='sampID', right_index=True)
                    filtered_df = staar_samples_df[staar_samples_df['FID'].astype(str).isin(pheno_null_model_samples)]

                    filtered_samples_path = Path(
                        f"{tarball_prefix.name}.{chromosome}.{phenoname}.STAAR.samples_table.tsv")
                    filtered_df.to_csv(filtered_samples_path, sep='\t', index=False)

                    link = exporter.export_files(filtered_samples_path)
                    pheno_filtered_samples[phenoname][tarball_prefix.name][chromosome] = link

        # 4. Run the actual per-gene association tests
        self._logger.info("Running STAAR masks across chromosomes...")
        launcher = joblauncher_factory(download_on_complete=True)
        valid_gene_ids = [
            gene_info.name
            for gene_infos in self.genetic_map.values() for gene_info in gene_infos
        ]

        # create export for transcript table
        transcripts_path = Path("transcripts_table_export.tsv")
        self._transcripts_table.to_csv(transcripts_path, sep='\t', index=True)
        transcripts_link = exporter.export_files(transcripts_path)

        for phenoname in self._association_pack.pheno_names:
            null_model_path = Path(f'{phenoname}.STAAR_null.rds')
            if not null_model_path.exists():
                raise FileNotFoundError(f"Null model for {phenoname} was not generated.")

            for tarball_prefix in self._association_pack.tarball_prefixes:
                for chromosome in self.genetic_map:
                    # If there were issues filtering, the key might not exist.
                    if chromosome not in pheno_filtered_samples[phenoname][tarball_prefix.name]:
                        continue

                    working_chunk = self._association_pack.bgen_dict[chromosome]

                    # Prepare file links
                    null_model_link = exporter.export_files(null_model_path)
                    variants_table_link = exporter.export_files(
                        f'{tarball_prefix}.{chromosome}.STAAR.variants_table.tsv')
                    filtered_samples_link = pheno_filtered_samples[phenoname][tarball_prefix.name][chromosome]

                    launcher.launch_job(
                        function=multithread_gene_model,
                        inputs={
                            'null_model': null_model_link,
                            'pheno_name': phenoname,
                            'tarball_prefix': tarball_prefix.name,
                            'chromosome': chromosome,
                            'genes': valid_gene_ids,
                            # CRITICAL: Send the link string, not the object
                            'bgen': working_chunk['bgen'].get_input_str(),
                            'index': working_chunk['index'].get_input_str(),
                            'sample': working_chunk['sample'].get_input_str(),
                            'filtered_samples': filtered_samples_link,
                            'staar_variants': variants_table_link,
                            # CRITICAL: Send string representation of enum
                            'tarball_type': str(self._association_pack.tarball_type),
                            'transcripts_table': transcripts_link
                        },
                        outputs=['output_model']
                    )

        launcher.submit_and_monitor()

        completed_staar_chunks = []
        for result in launcher:
            # Launcher automatically downloads outputs to local path
            output_path = InputFileHandler(result['output_model']).get_file_handle()
            df = pd.read_csv(output_path, sep='\t', index_col=0)
            completed_staar_chunks.append(df)

        if completed_staar_chunks:
            combined_staar = pd.concat(completed_staar_chunks, axis=0)
            combined_staar = combined_staar.sort_values(by='pheno_name')

            # Write to file
            output_tsv = Path(f"{self._output_prefix}.genes.STAAR.stats.tsv")
            combined_staar.to_csv(output_tsv, sep='\t', index=True)
            outputs = bgzip_and_tabix(output_tsv, comment_char="E", skip_row=1, sequence_row=14, begin_row=15,
                                      end_row=16, force=True)
            self._outputs.extend(outputs)
            self._logger.info("STAAR stats written to %s.genes.STAAR.stats.tsv", self._output_prefix)


def _process_staar_gene(gene: str, gene_data: dict, bgen_path: Path, sample_path: Path, keep_rows: List[int],
                        tarball_prefix: str, chromosome: str, pheno_name: str, null_model: Path,
                        filtered_samples_path: Path, staar_variants: Path) -> dict:
    """Helper function to process a single gene for STAAR analysis.

    This function generates the CSR matrix for a given gene, subsets it based on the provided samples,
    and returns a dictionary of parameters for launching a `staar_genes` job.

    :param gene: The ENST ID of the gene to process.
    :param gene_data: A dictionary of data for this gene (variants, coordinates, etc.).
    :param bgen_path: Path to the BGEN file.
    :param sample_path: Path to the BGEN sample file.
    :param keep_rows: A list of row indices to keep from the genotype matrix.
    :param tarball_prefix: The prefix of the tarball.
    :param chromosome: The chromosome being processed.
    :param pheno_name: The name of the phenotype.
    :param null_model: Path to the STAAR null model file.
    :param filtered_samples_path: Path to the filtered STAAR samples file.
    :param staar_variants: Path to the STAAR variants table.
    :return: A dictionary of parameters for `thread_utility.launch_job`.
    """
    matrix, _ = generate_csr_matrix_from_bgen(
        bgen_path=bgen_path,
        sample_path=sample_path,
        variant_filter_list=gene_data['vars'],
        chromosome=gene_data['chrom'],
        start=gene_data['min'],
        end=gene_data['max'],
        should_collapse_matrix=False
    )
    matrix = matrix[keep_rows, :]
    matrix_path = f"{tarball_prefix}.{chromosome}.{gene}.STAAR.mtx"
    mmwrite(matrix_path, matrix)

    return {
        'function': staar_genes,
        'inputs': {
            'staar_null_path': null_model,
            'pheno_name': pheno_name,
            'gene': gene,
            'mask_name': tarball_prefix,
            'staar_matrix': matrix_path,
            'staar_samples': filtered_samples_path,
            'staar_variants': staar_variants,
            'out_dir': Path('.'),
        },
        'outputs': ['staar_result']
    }


@dxpy.entry_point('multithread_gene_model')
def multithread_gene_model(null_model: str, pheno_name: str, tarball_prefix: str, chromosome: str, genes: List[str],
                           bgen: str, index: str, sample: str,
                           filtered_samples: str, staar_variants: str, tarball_type: str,
                           transcripts_table: str) -> Dict[str, str]:
    """
    Run a STAAR gene model in a multithreaded way for a single chromosome.

    :param null_model: DNAnexus file-ID of the null model RDS file.
    :param pheno_name: The phenotype name.
    :param tarball_prefix: The tarball prefix to work with.
    :param chromosome: The chromosome to process.
    :param genes: A list of gene ENST IDs to run.
    :param bgen: DNAnexus file-ID for the BGEN file.
    :param index: DNAnexus file-ID for the BGEN index file.
    :param sample: DNAnexus file-ID for the BGEN sample file.
    :param filtered_samples: DNAnexus file-ID for the pre-filtered STAAR samples table.
    :param staar_variants: DNAnexus file-ID for the STAAR variants table.
    :param tarball_type: The tarball type (string representation of TarballType enum).
    :param transcripts_table: DNAnexus file-ID for the transcripts table.
    :return: A dictionary containing the DNAnexus file-ID of the output STAAR results TSV.
    """

    # 1. SETUP & DOWNLOAD
    null_model_path = InputFileHandler(null_model).get_file_handle()
    filtered_samples_path = InputFileHandler(filtered_samples).get_file_handle()
    staar_variants_path = InputFileHandler(staar_variants).get_file_handle()
    transcripts_table_path = InputFileHandler(transcripts_table).get_file_handle()
    bgen_path = InputFileHandler(bgen).get_file_handle()
    _ = InputFileHandler(index).get_file_handle()
    sample_path = InputFileHandler(sample).get_file_handle()

    # 2. LOAD STAAR DATA & SAMPLES FOR FILTERING
    staar_data = load_staar_genetic_data(tarball_prefix, chromosome)
    filtered_samples_df = pd.read_csv(filtered_samples_path, sep='\t')
    keep_rows = filtered_samples_df['row'].values.tolist()

    # 3. PROCESS GENES IN PARALLEL
    thread_utility = ThreadUtility()
    valid_gene_ids = set(genes)

    genes_per_chunk = {
        chunk: [gene for gene in chunk_genes.keys() if gene in valid_gene_ids]
        for chunk, chunk_genes in staar_data.items()
    }

    for chunk, gene_list in genes_per_chunk.items():
        if not gene_list:
            continue
        for gene in gene_list:
            job_params = _process_staar_gene(
                gene=gene,
                gene_data=staar_data[chunk][gene],
                bgen_path=bgen_path,
                sample_path=sample_path,
                keep_rows=keep_rows,
                tarball_prefix=tarball_prefix,
                chromosome=chromosome,
                pheno_name=pheno_name,
                null_model=null_model_path,
                filtered_samples_path=filtered_samples_path,
                staar_variants=staar_variants_path
            )
            thread_utility.launch_job(**job_params)

    thread_utility.submit_and_monitor()

    # 4. COLLECT RESULTS & ANNOTATE
    completed_staar_files = [result["staar_result"] for result in thread_utility]

    if completed_staar_files:
        transcript_df = pd.read_csv(transcripts_table_path, sep='\t', index_col=0)
        output_model = Path(f'{pheno_name}.{chromosome}.staar_results.tsv')

        process_model_outputs(input_models=completed_staar_files,
                              output_path=output_model,
                              tarball_type=tarball_type,
                              transcripts_table=transcript_df)

        # 5. EXPORT & RETURN
        exporter = ExportFileHandler()
        uploaded_file = exporter.export_files(output_model)

        return {"output_model": uploaded_file}
    else:
        # Need to return something...
        return {"output_model": ""}