import csv
import json
from pathlib import Path
from typing import List

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

        if self._gene_infos:
            with open('staar.gene_list', 'w') as gene_list_file:
                for gene_info in self._gene_infos:
                    gene_list_file.write(f"{gene_info.name}\n")

        # 1. Run the STAAR NULL model
        self._logger.info("Running STAAR Null Model(s)...")
        thread_utility = ThreadUtility(self._association_pack.threads, thread_factor=1)
        for phenoname in self._association_pack.pheno_names:
            thread_utility.launch_job(function=staar_null,
                                      inputs={
                                          'phenofile': self._association_pack.final_covariates,
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

        # 2. Run the actual per-gene association tests
        self._logger.info("Running STAAR masks across chromosomes...")
        launcher = joblauncher_factory(download_on_complete=True)

        for phenoname in self._association_pack.pheno_names:
            for tarball_prefix in self._association_pack.tarball_prefixes:
                for chromosome in self._chromosomes:
                    staar_data = load_staar_genetic_data(
                        tarball_prefix=tarball_prefix,
                        bgen_prefix=chromosome
                    )

                    valid_gene_ids = {gene.name for gene in self._gene_infos}
                    genes_per_chunk = {
                        chunk: [gene for gene in genes.keys() if gene in valid_gene_ids]
                        for chunk, genes in staar_data.items()
                    }

                    for chunk, gene_list in genes_per_chunk.items():
                        if not gene_list:
                            continue

                        subset_staar_data = {chunk: staar_data[chunk]}
                        chunk_json_path = Path(f"{tarball_prefix}.{chunk}.staar_chunk.json")
                        with chunk_json_path.open("w") as f:
                            json.dump(subset_staar_data, f, default=lambda o: list(o) if isinstance(o, set) else o)

                        working_chunk = self._association_pack.bgen_dict[chromosome]

                        exporter = ExportFileHandler(delete_on_upload=False)
                        null_model = exporter.export_files(f'{phenoname}.STAAR_null.rds')
                        staar_samples = exporter.export_files(f'{tarball_prefix}.{chromosome}.STAAR.samples_table.tsv')
                        variants_table = exporter.export_files(
                            f'{tarball_prefix}.{chromosome}.STAAR.variants_table.tsv')
                        chunk_file = exporter.export_files(chunk_json_path)
                        transcripts_table = Path("transcripts_table.tsv")
                        self._transcripts_table.to_csv(transcripts_table, sep='\t', index=True)
                        transcripts_table = exporter.export_files(transcripts_table)

                        launcher.launch_job(
                            function=multithread_gene_model,
                            inputs={
                                'null_model': null_model,
                                'pheno_name': phenoname,
                                'tarball_prefix': tarball_prefix,
                                'chromosome': chromosome,
                                'genes': gene_list,
                                'chunk_file': chunk_file,
                                'bgen': working_chunk['bgen'],
                                'index': working_chunk['index'],
                                'sample': working_chunk['sample'],
                                'staar_samples': staar_samples,
                                'staar_variants': variants_table,
                                'tarball_type': self._association_pack.tarball_type,
                                'transcripts_table': transcripts_table
                            },
                            outputs=['output_model']
                        )

        launcher.submit_and_monitor()

        completed_staar_chunks = []
        for result in launcher:
            df = pd.read_csv(result['output_model'], sep='\t', index_col=0)
            completed_staar_chunks.append(df)

        if completed_staar_chunks:
            combined_staar = pd.concat(completed_staar_chunks, axis=0).sort_values(by='start')
            combined_staar.to_csv(f'{self._output_prefix}.genes.STAAR.stats.tsv', sep='\t', index=True)
            output_tsv = Path(f"{self._output_prefix}.genes.STAAR.stats.tsv")
            outputs = bgzip_and_tabix(output_tsv, skip_row=1, sequence_row=2, begin_row=3, end_row=4)
            self._outputs.extend(outputs)
            self._logger.info("STAAR stats written to %s.genes.STAAR.stats.tsv", self._output_prefix)
        else:
            self._logger.warning("No STAAR results generated.")


@dxpy.entry_point('multithread_gene_model')
def multithread_gene_model(null_model, pheno_name, tarball_prefix, chromosome, genes, chunk_file, bgen, index, sample,
                           staar_samples, staar_variants, tarball_type, transcripts_table) -> Path:
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

    # load our VM environment
    _ = build_default_command_executor()
    null_model = InputFileHandler(null_model).get_file_handle()
    staar_samples = InputFileHandler(staar_samples).get_file_handle()
    staar_variants = InputFileHandler(staar_variants).get_file_handle()
    chunk_file = InputFileHandler(chunk_file).get_file_handle()
    transcripts_table = InputFileHandler(transcripts_table).get_file_handle()

    with open(chunk_file, "r") as f:
        staar_data = json.load(f)

    # download our bgen files
    bgen_path = bgen.get_file_handle()
    _ = index.get_file_handle()
    sample_path = sample.get_file_handle()

    thread_utility = ThreadUtility()

    for gene in genes:
        # generate a csr matrix from the bgen files
        matrix, _ = generate_csr_matrix_from_bgen(
            bgen_path=bgen_path,
            sample_path=sample_path,
            variant_filter_list=staar_data[chromosome][gene]['vars'],
            chromosome=staar_data[chromosome][gene]['chrom'],
            start=staar_data[chromosome][gene]['min'],
            end=staar_data[chromosome][gene]['max'],
            should_collapse_matrix=False
        )

        # export matrix to file
        mmwrite(f"{tarball_prefix}.{chromosome}.STAAR.mtx", matrix)

        thread_utility.launch_job(
            function=staar_genes,
            inputs={
                'staar_null_path': null_model,
                'pheno_name': pheno_name,
                'gene': gene,  # single ENST ID string
                'mask_name': tarball_prefix,
                'staar_matrix': f"{tarball_prefix}.{chromosome}.STAAR.mtx",
                'staar_samples': staar_samples,
                'staar_variants': staar_variants,
                'out_dir': Path('.'),
            },
            outputs=['staar_result']
        )
    thread_utility.submit_and_monitor()
    # Print a preliminary STAAR output
    completed_staar_files = []
    # And gather the resulting futures
    for result in thread_utility:
        # Each result is a dict with {'staar_result': STAARModelResult(...)}
        staar_result = result["staar_result"]
        completed_staar_files.append(staar_result)

    # Annotate STAAR output
    transcript = pd.read_csv(transcripts_table, sep='\t', index_col=0)
    output_model = Path(f'{chromosome}.staar_results.tsv')
    process_model_outputs(input_models=completed_staar_files,
                          output_path=output_model,
                          tarball_type=tarball_type,
                          transcripts_table=transcript)

    return output_model
