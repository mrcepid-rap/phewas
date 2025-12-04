import csv
import json
from pathlib import Path
from typing import List, Dict

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
from general_utilities.job_management.command_executor import build_default_command_executor
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
        self._chromosomes = set()

        # build the transcripts table
        self._transcripts_table = build_transcript_table(transcripts_path=self._association_pack.transcript_index)

        # Figure out genes/SNPlist to run...
        self._gene_infos = []
        if self._association_pack.is_non_standard_tar:
            gene_info, returned_chromosomes = process_snp_or_gene_tar(self._association_pack.is_snp_tar,
                                                                      self._association_pack.is_gene_tar,
                                                                      self._association_pack.tarball_prefixes[0])
            self._gene_infos.append(gene_info)
            self._chromosomes = returned_chromosomes
        else:
            for gene in self._association_pack.gene_ids:
                gene_info = get_gene_id(gene, self._transcripts_table)
                self._gene_infos.append(gene_info)

                for chunk in self._association_pack.bgen_dict:
                    chromosomes = process_gene_or_snp_wgs(
                        identifier=gene_info.name,
                        tarball_prefix=self._association_pack.tarball_prefixes[0],
                        chunk=chunk
                    )

                    if chromosomes:
                        self._logger.info(f"{gene_info['SYMBOL']} found in {chunk} ({', '.join(chromosomes)})")
                        self._chromosomes.add(chunk)

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
                self._association_pack.is_snp_tar,
                self._association_pack.is_gene_tar,
            )
        )

    def _run_linear_models(self) -> None:
        """
        Run linear models for each gene and chromosome combination.
        """
        null_model = linear_model.linear_model_null(
            phenotype=self._association_pack.pheno_names[0],
            phenofile=self._association_pack.final_covariates,
            is_binary=self._association_pack.is_binary,
            ignore_base=self._association_pack.ignore_base_covariates,
            found_quantitative_covariates=self._association_pack.found_quantitative_covariates,
            found_categorical_covariates=self._association_pack.found_categorical_covariates
        )

        self._logger.info("Loading linear model genotypes...")
        thread_utility = ThreadUtility(self._association_pack.threads, thread_factor=2)
        for chromosome in self._chromosomes:
            for tarball_prefix in self._association_pack.tarball_prefixes:
                thread_utility.launch_job(
                    function=linear_model.load_linear_model_genetic_data,
                    inputs={
                        'tarball_prefix': tarball_prefix,
                        'tarball_type': self._association_pack.tarball_type,
                        'bgen_prefix': chromosome,
                    },
                    outputs=['tarball_prefix', 'genetic_data']
                )
        thread_utility.submit_and_monitor()

        genotype_packs = {}
        for result in thread_utility:
            tarball_prefix = result['tarball_prefix']
            genotype_packs[tarball_prefix] = result['genetic_data']

        thread_utility = ThreadUtility(self._association_pack.threads, thread_factor=1)
        for model in genotype_packs:
            for gene_info in self._gene_infos:
                thread_utility.launch_job(
                    function=linear_model.run_linear_model,
                    inputs={
                        'linear_model_pack': null_model,
                        'genotype_table': genotype_packs[model],
                        'gene': gene_info.name,
                        'mask_name': model,
                        'is_binary': self._association_pack.is_binary,
                        'always_run_corrected': True
                    },
                    outputs=['gene_dict']
                )
        thread_utility.submit_and_monitor()

        fieldnames = ['ENST', 'mask_name', 'pheno_name', 'p_val_init', 'n_car', 'cMAC', 'n_model',
                      'p_val_full', 'effect', 'std_err']
        if self._association_pack.is_binary:
            fieldnames.extend(['n_noncar_affected', 'n_noncar_unaffected', 'n_car_affected', 'n_car_unaffected'])

        lm_stats_path = Path(f'{self._output_prefix}.lm_stats.tmp')
        with lm_stats_path.open('w') as lm_stats_file:
            lm_stats_writer = csv.DictWriter(lm_stats_file,
                                             delimiter="\t",
                                             fieldnames=fieldnames,
                                             extrasaction='ignore')
            lm_stats_writer.writeheader()
            finished_genes = []
            for result in thread_utility:
                finished_gene: LinearModelResult = result['gene_dict']
                lm_stats_writer.writerow(finished_gene.todict())
                finished_genes.append(finished_gene)

        process_model_outputs(input_models=finished_genes,
                              output_path=Path(f'{self._output_prefix}.genes.glm.stats.tsv'),
                              tarball_type=self._association_pack.tarball_type,
                              transcripts_table=self._transcripts_table)
        self._logger.info("Linear model stats written to %s.genes.glm.stats.tsv", self._output_prefix)

    def _run_staar_models(self):
        """
        Run STAAR models for each gene and chromosome combination.
        """

        # 0. Create the gene list for the R script
        if self._gene_infos:
            with open('staar.gene_list', 'w') as gene_list_file:
                for gene_info in self._gene_infos:
                    gene_list_file.write(f"{gene_info.name}\n")

        self._logger.info("Creating merged covariates file for STAAR null model...")
        # Grab the first chromosome's sample file just to get the IDs
        first_chrom = next(iter(self._chromosomes))
        sample_path = self._association_pack.bgen_dict[first_chrom]["sample"].get_file_handle()

        # Read sample file (BGEN format usually has 2 header rows, we skip the second type row)
        sample = pd.read_csv(sample_path, sep=r"\s+", header=0, dtype={'ID_2': str})
        sample = sample.drop(columns=["sex"], errors="ignore")
        sample = sample.iloc[1:].reset_index(drop=True)  # Drop the "0 0 0 D D D..." row

        # Read covariates
        covar = pd.read_csv(self._association_pack.final_covariates, sep=' ', header=0, dtype={'IID': str})

        # Merge BGEN IDs with Covariates
        merged = sample.merge(covar, how="left", left_on="ID_2", right_on="IID")
        merged = merged.drop(columns=["ID_1", "missing"], errors="ignore")
        merged = merged.sort_values("ID_2")

        # Remove rows with missing phenotype data
        for phenoname in self._association_pack.pheno_names:
            if phenoname in merged.columns:
                merged = merged.dropna(subset=[phenoname])

        # Remove rows with missing covariate data
        required_cols = ['age', 'age_squared', 'batch']
        if self._association_pack.sex == 2:
            required_cols.append('sex')
        for i in range(1, 11):
            required_cols.append(f'PC{i}')
        required_cols.extend(self._association_pack.found_quantitative_covariates)
        required_cols.extend(self._association_pack.found_categorical_covariates)

        # Only check columns that actually exist
        required_cols = [col for col in required_cols if col in merged.columns]
        merged = merged.dropna(subset=required_cols)

        self._logger.info(f"After removing samples with missing data: {len(merged)} samples remain")

        # Save this 'clean' file for the Null Model
        merged = merged.drop(columns=['IID', 'FID'], errors='ignore')
        merged = merged.rename(columns={'ID_2': 'FID'})
        merged_cov_path = Path("merged_covariates_for_staar.tsv")
        merged.to_csv(merged_cov_path, sep="\t", index=False)

        # 1. Run the STAAR NULL model
        self._logger.info("Running STAAR Null Model(s)...")
        thread_utility = ThreadUtility(self._association_pack.threads, thread_factor=1)

        for phenoname in self._association_pack.pheno_names:
            # We wrap staar_null or define a helper to return (pheno, path)
            # using the NEW merged_cov_path
            thread_utility.launch_job(
                function=staar_null,
                inputs={
                    'phenofile': merged_cov_path,  # Use the clean file!
                    'phenotype': phenoname,
                    'is_binary': self._association_pack.is_binary,
                    'ignore_base': self._association_pack.ignore_base_covariates,
                    'found_quantitative_covariates': self._association_pack.found_quantitative_covariates,
                    'found_categorical_covariates': self._association_pack.found_categorical_covariates,
                    'sex': self._association_pack.sex,
                    'sparse_kinship_file': self._association_pack.sparse_grm,
                    'sparse_kinship_samples': self._association_pack.sparse_grm_sample
                }
            )
        thread_utility.submit_and_monitor()

        # Collect the null models.
        # Note: staar_null returns a Path. We need to map it back to the phenotype.
        # Assuming staar_null outputs a file named '{phenotype}.STAAR_null.rds' locally.
        null_models = {}
        for phenoname in self._association_pack.pheno_names:
            expected_path = Path(f'{phenoname}.STAAR_null.rds')
            if expected_path.exists():
                null_models[phenoname] = expected_path
            else:
                raise FileNotFoundError(f"Null model for {phenoname} was not generated.")

        null_model_samples = set(merged['FID'].astype(str))

        for tarball_prefix in self._association_pack.tarball_prefixes:
            for chromosome in self._chromosomes:
                samples_path = Path(f"{tarball_prefix}.{chromosome}.STAAR.samples_table.tsv")

                if samples_path.exists():
                    staar_samples_df = pd.read_csv(samples_path, sep='\t')
                    # Keep only samples present in the Null Model
                    staar_samples_df = staar_samples_df[
                        staar_samples_df['sampID'].astype(str).isin(null_model_samples)
                    ]
                    staar_samples_df.to_csv(samples_path, sep='\t', index=False)

        # 2. Run the actual per-gene association tests
        self._logger.info("Running STAAR masks across chromosomes...")
        launcher = joblauncher_factory(download_on_complete=True)

        for phenoname in self._association_pack.pheno_names:
            for tarball_prefix in self._association_pack.tarball_prefixes:
                for chromosome in self._chromosomes:

                    # ... (Existing code to load staar_data and valid_gene_ids) ...
                    staar_data = load_staar_genetic_data(tarball_prefix, chromosome)
                    valid_gene_ids = {gene.name for gene in self._gene_infos}

                    genes_per_chunk = {
                        chunk: [gene for gene in genes.keys() if gene in valid_gene_ids]
                        for chunk, genes in staar_data.items()
                    }

                    for chunk, gene_list in genes_per_chunk.items():
                        if not gene_list: continue

                        # Dump chunk json
                        subset_staar_data = {chunk: staar_data[chunk]}
                        chunk_json_path = Path(f"{tarball_prefix}.{chunk}.staar_chunk.json")
                        with chunk_json_path.open("w") as f:
                            json.dump(subset_staar_data, f, default=lambda o: list(o) if isinstance(o, set) else o)

                        working_chunk = self._association_pack.bgen_dict[chromosome]
                        exporter = ExportFileHandler(delete_on_upload=False)

                        # Prepare file links
                        null_model_link = exporter.export_files(null_models[phenoname])
                        staar_samples_link = exporter.export_files(
                            f'{tarball_prefix}.{chromosome}.STAAR.samples_table.tsv')
                        variants_table_link = exporter.export_files(
                            f'{tarball_prefix}.{chromosome}.STAAR.variants_table.tsv')
                        chunk_file_link = exporter.export_files(chunk_json_path)

                        transcripts_path = Path("transcripts_table.tsv")
                        self._transcripts_table.to_csv(transcripts_path, sep='\t', index=True)
                        transcripts_link = exporter.export_files(transcripts_path)

                        launcher.launch_job(
                            function=multithread_gene_model,
                            inputs={
                                'null_model': null_model_link,
                                'pheno_name': phenoname,
                                'tarball_prefix': tarball_prefix,
                                'chromosome': chromosome,
                                'genes': gene_list,
                                'chunk_file': chunk_file_link,
                                # CRITICAL: Send the link string, not the object
                                'bgen': working_chunk['bgen'].get_input_str(),
                                'index': working_chunk['index'].get_input_str(),
                                'sample': working_chunk['sample'].get_input_str(),
                                'staar_samples': staar_samples_link,
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

            # Merge with transcripts table to get coordinates
            # We must do this BEFORE sorting
            combined_staar = combined_staar.merge(
                self._transcripts_table[['chrom', 'start', 'end']],
                left_index=True,
                right_index=True,
                how='left'
            )
            combined_staar = combined_staar.sort_values(by='start')
            # Write to file
            output_tsv = Path(f"{self._output_prefix}.genes.STAAR.stats.tsv")
            combined_staar.to_csv(output_tsv, sep='\t', index=True)
            outputs = bgzip_and_tabix(output_tsv, skip_row=1, sequence_row=14, begin_row=15, end_row=16)
            self._outputs.extend(outputs)
            self._logger.info("STAAR stats written to %s.genes.STAAR.stats.tsv", self._output_prefix)
        else:
            self._logger.warning("No STAAR results generated.")


@dxpy.entry_point('multithread_gene_model')
def multithread_gene_model(null_model, pheno_name, tarball_prefix, chromosome, genes, chunk_file, bgen, index, sample,
                           staar_samples, staar_variants, tarball_type, transcripts_table) -> Dict[str, str]:
    """
    Run a STAAR gene model in a multithreaded way

    :param null_model: a path to the null model RDS file
    :param pheno_name: the phenotype name
    :param tarball_prefix: the tarball prefix to work with
    :param chromosome: the chromosome chunk to work with
    :param genes: list of genes to run
    :param chunk_file: a path to the chunk JSON file (contains the genetic coordinates & variants)
    :param bgen: InputFileHandler for the bgen file
    :param index: InputFileHandler for the bgen index file
    :param sample: InputFileHandler for the bgen sample file
    :param staar_samples: STAAR samples table for the chunk we are working with
    :param staar_variants: STAAR variants table for the chunk we are working with
    :param tarball_type: the tarball type (TarballType enum)
    :param transcripts_table: a path to the transcripts table
    :return: Path to the output STAAR results TSV file (post-annotation)
    """

    # 1. SETUP & DOWNLOAD
    # We must rebuild the command executor and file handlers
    _ = build_default_command_executor()

    # Download inputs using InputFileHandler
    null_model = InputFileHandler(null_model).get_file_handle()
    staar_samples = InputFileHandler(staar_samples).get_file_handle()
    staar_variants = InputFileHandler(staar_variants).get_file_handle()
    chunk_file = InputFileHandler(chunk_file).get_file_handle()
    transcripts_table = InputFileHandler(transcripts_table).get_file_handle()

    # BGEN files need download_now=True usually, or just get_file_handle
    bgen_path = InputFileHandler(bgen).get_file_handle()
    _ = InputFileHandler(index).get_file_handle()
    sample_path = InputFileHandler(sample).get_file_handle()

    with open(chunk_file, "r") as f:
        staar_data = json.load(f)

    # 2. LOAD SAMPLES FOR FILTERING
    # This file was filtered in the main class to only include IDs in the Null Model.
    # We need the 'row' indices to subset the BGEN matrix.
    staar_samples_df = pd.read_csv(staar_samples, sep='\t')
    keep_rows = staar_samples_df['row'].values

    thread_utility = ThreadUtility()

    for gene in genes:
        # Check if gene is valid in this chunk
        if gene not in staar_data[chromosome]:
            continue

        # 3. GENERATE & SUBSET MATRIX
        # Generate full matrix
        matrix, _ = generate_csr_matrix_from_bgen(
            bgen_path=bgen_path,
            sample_path=sample_path,
            variant_filter_list=staar_data[chromosome][gene]['vars'],
            chromosome=staar_data[chromosome][gene]['chrom'],
            start=staar_data[chromosome][gene]['min'],
            end=staar_data[chromosome][gene]['max'],
            should_collapse_matrix=False
        )

        # CRITICAL FIX: Subset matrix to match Null Model dimensions
        matrix = matrix[keep_rows, :]

        # Save the SUBSETTED matrix
        mmwrite(f"{tarball_prefix}.{chromosome}.STAAR.mtx", matrix)

        thread_utility.launch_job(
            function=staar_genes,
            inputs={
                'staar_null_path': null_model,
                'pheno_name': pheno_name,
                'gene': gene,
                'mask_name': tarball_prefix,
                'staar_matrix': f"{tarball_prefix}.{chromosome}.STAAR.mtx",
                'staar_samples': staar_samples,
                'staar_variants': staar_variants,
                'out_dir': Path('.'),
            },
            outputs=['staar_result']
        )
    thread_utility.submit_and_monitor()

    completed_staar_files = []
    for result in thread_utility:
        staar_result = result["staar_result"]
        completed_staar_files.append(staar_result)

    # Annotate STAAR output
    transcript = pd.read_csv(transcripts_table, sep='\t', index_col=0)
    output_model = Path(f'{chromosome}.staar_results.tsv')

    process_model_outputs(input_models=completed_staar_files,
                          output_path=output_model,
                          tarball_type=tarball_type,
                          transcripts_table=transcript)

    # 4. EXPORT & RETURN
    # DX entry points must return a dictionary of links, not a Path object
    exporter = ExportFileHandler()
    uploaded_file = exporter.export_files(output_model)

    return {"output_model": uploaded_file}
