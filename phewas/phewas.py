import csv
import numpy as np
import pandas as pd
import statsmodels.api as sm

from typing import List
from pathlib import Path
from statsmodels.tools.sm_exceptions import PerfectSeparationError


from general_utilities.association_resources import process_snp_or_gene_tar, build_transcript_table, get_gene_id
from general_utilities.linear_model.linear_model import load_tarball_linear_model
from general_utilities.linear_model.proccess_model_output import process_linear_model_outputs, merge_glm_staar_runs, \
    process_staar_outputs
from general_utilities.linear_model.staar_model import staar_null, staar_genes
from general_utilities.job_management.thread_utility import ThreadUtility
from phewas.phewas_association_pack import PhewasAssociationPack


class PheWAS:

    def __init__(self, output_prefix: str, association_pack: PhewasAssociationPack):

        self._association_pack = association_pack
        self._output_prefix = output_prefix
        self._outputs = []

        # Figure out genes/SNPlist to run...
        self._gene_infos = []
        if self._association_pack.is_non_standard_tar:
            gene_info, returned_chromosomes = process_snp_or_gene_tar(self._association_pack.is_snp_tar,
                                                                      self._association_pack.is_gene_tar,
                                                                      self._association_pack.tarball_prefixes[0])
            self._gene_infos.append(gene_info)
        else:
            for gene in self._association_pack.gene_ids:
                transcripts_table = build_transcript_table()
                gene_info = get_gene_id(gene, transcripts_table)
                self._gene_infos.append(gene_info)

    def _add_output(self, file: Path) -> None:
        self._outputs.append(file)

    def _extend_output(self, files: List[Path]) -> None:
        self._outputs.extend(files)

    def get_outputs(self) -> List[Path]:
        return self._outputs

    def run_tool(self):

        self._add_output(Path('phenotypes_covariates.formatted.txt'))

        # 2. Load the tarballs into separate genotypes dictionaries
        print("Loading Linear Model genotypes")
        thread_utility = ThreadUtility(self._association_pack.threads,
                                       error_message='A GLM thread failed',
                                       incrementor=10,
                                       thread_factor=2)

        for tarball_prefix in self._association_pack.tarball_prefixes:
            thread_utility.launch_job(load_tarball_linear_model,
                                      tarball_prefix=tarball_prefix,
                                      is_snp_tar=self._association_pack.is_snp_tar,
                                      is_gene_tar=self._association_pack.is_gene_tar)

        genotype_packs = {}
        for result in thread_utility:
            tarball_prefix, genotype_dict = result
            genotype_packs[tarball_prefix] = genotype_dict

        # 3. Add a variable for all gene/mask combinations the phenotype/covariate pandas data frame.
        model_dictionary = self._build_model_dictionary(genotype_packs, self._gene_infos)

        # 4. Now run the actual models in parallel
        # I think this is straight-forward?
        print("Submitting linear models...")

        thread_utility = ThreadUtility(self._association_pack.threads,
                                       error_message='A GLM thread failed',
                                       incrementor=100,
                                       thread_factor=2)

        for phenoname in self._association_pack.pheno_names:

            # Set the base formula for this pheno
            base_formula = self._set_base_formula(phenoname)

            for model in genotype_packs:
                for gene in self._gene_infos:
                    formatted_formula = f'{base_formula} + {self._set_var_name(model, gene.name)}'
                    thread_utility.launch_job(self._linear_model_phewas,
                                              model_dictionary=model_dictionary,
                                              formula=formatted_formula,
                                              phenoname=phenoname,
                                              gene=gene.name,
                                              mask_name=model)

        # Here we're collecting futures and writing the unformatted results at the same time
        fieldnames = ['ENST', 'maskname', 'pheno_name', 'p_val_init', 'n_car', 'cMAC', 'n_model',
                      'p_val_full', 'effect', 'std_err', 'model_run']
        if self._association_pack.is_binary:
            fieldnames.extend(['n_noncar_affected', 'n_noncar_unaffected', 'n_car_affected', 'n_car_unaffected'])

        lm_stats_file = open(self._output_prefix + '.lm_stats.tmp', 'w')
        lm_stats_writer = csv.DictWriter(lm_stats_file,
                                         delimiter="\t",
                                         fieldnames=fieldnames)

        lm_stats_writer.writeheader()
        for result in thread_utility:
            finished_gene = result
            lm_stats_writer.writerow(finished_gene)
        lm_stats_file.close()

        # 5. Run a STAAR model for each phenotype:
        print("Running STAAR models...")
        self._run_staar_models()

        # 6. Annotate unformatted results and print final outputs
        print("Annotating Linear Model results")
        process_linear_model_outputs(self._output_prefix, self._association_pack.is_snp_tar,
                                     self._association_pack.is_gene_tar, self._gene_infos)

        # 7. Merging GLM/STAAR runs:
        self._extend_output(merge_glm_staar_runs(self._output_prefix, self._association_pack.is_snp_tar,
                                                 self._association_pack.is_gene_tar))

    # I think this is the most efficient way to do this on a large memory machine where I can store everything in
    # one massive DataFrame. Everything else I have tried takes a very long time!
    @staticmethod
    def _build_model_dictionary(genotype_packs: dict, gene_infos: list) -> pd.DataFrame:

        # First load the model dictionary
        model_dictionary = pd.read_csv("phenotypes_covariates.formatted.txt",
                                       sep=" ",
                                       index_col="FID",
                                       dtype={'IID': str})
        model_dictionary.index = model_dictionary.index.astype(str)

        # And then iterate through all possible tarball/gene combinations and add a column for mask+gene pairs
        for model in genotype_packs:
            for gene in gene_infos:
                indv_w_var = genotype_packs[model].loc[gene.name]
                var_name = '%s_%s' % (model, gene.name)
                # Need to reserved characters for formula writing from the
                # var_name string (I think this is all of them...):
                var_name = var_name.translate(
                    str.maketrans({'-': '_', '+': '_', '(': '_', ')': '_', '~': '_', '*': '_'}))
                model_dictionary = pd.merge(model_dictionary, indv_w_var, how='left', on='FID')
                model_dictionary[var_name] = model_dictionary['gt'].transform(
                    lambda x: 0 if np.isnan(x) else x)
                model_dictionary = model_dictionary.drop(columns=['gt'])

        return model_dictionary

    def _set_base_formula(self, phenoname: str) -> str:
        covars = ['sex', 'age', 'age_squared', 'wes_batch'] + ['PC' + str(PC) for PC in range(1, 11)]

        if len(self._association_pack.found_quantitative_covariates) > 0:
            for covar in self._association_pack.found_quantitative_covariates:
                covars.append(covar)
        if len(self._association_pack.found_categorical_covariates) > 0:
            for covar in self._association_pack.found_categorical_covariates:
                covars.append('C(' + covar + ')')

        base_formula = phenoname + ' ~ ' + ' + '.join(covars)

        return base_formula

    @staticmethod
    def _set_var_name(model: str, gene: str) -> str:
        var_name = '%s_%s' % (model, gene)
        # Need to reserved characters for formula writing from the string name:
        var_name = var_name.translate(
            str.maketrans({'-': '_', '+': '_', '(': '_', ')': '_', '~': '_', '*': '_'}))
        return var_name

    def _linear_model_phewas(self, model_dictionary: pd.DataFrame, formula: str, phenoname: str,
                             gene: str, mask_name: str) -> dict:

        id_column = self._set_var_name(mask_name, gene)

        # Calculate Number of Carriers
        n_car = len(model_dictionary.loc[model_dictionary[id_column] == 1])
        cMAC = model_dictionary[id_column].sum()

        # Default return:
        gene_dict = {'p_val_init': 'NA',
                     'n_car': n_car,
                     'cMAC': cMAC,
                     'n_model': len(model_dictionary['IID']),
                     'ENST': gene,
                     'maskname': mask_name,
                     'pheno_name': phenoname,
                     'p_val_full': 'NA',
                     'effect': 'NA',
                     'std_err': 'NA',
                     'model_run': 'false'}
        if self._association_pack.is_binary:
            gene_dict['n_noncar_affected'] = len(model_dictionary.query(f'{id_column} == 0 & {phenoname} == 1'))
            gene_dict['n_noncar_unaffected'] = len(model_dictionary.query(f'{id_column} == 0 & {phenoname} == 0'))
            gene_dict['n_car_affected'] = len(model_dictionary.query(f'{id_column} == 1 & {phenoname} == 1'))
            gene_dict['n_car_unaffected'] = len(model_dictionary.query(f'{id_column} == 1 & {phenoname} == 0'))

        if n_car <= 2 or (self._association_pack.is_binary and (gene_dict['n_car_affected'] <= 2)):
            return gene_dict
        else:
            try:
                sm_results = sm.GLM.from_formula(formula,
                                                 data=model_dictionary,
                                                 family=sm.families.Binomial() if
                                                 self._association_pack.is_binary else sm.families.Gaussian(),
                                                 missing='drop').fit()

                gene_dict['p_val_init'] = sm_results.pvalues[id_column]
                gene_dict['p_val_full'] = sm_results.pvalues[id_column]
                gene_dict['effect'] = sm_results.params[id_column]
                gene_dict['std_err'] = sm_results.bse[id_column]
                gene_dict['model_run'] = 'true'

                return gene_dict

            except PerfectSeparationError:

                return gene_dict

    def _run_staar_models(self):

        valid_staar_chromosomes = set()

        # Create a file of genes if genes_to_run is !none
        if self._gene_infos is not None:
            with open('staar.gene_list', 'w') as gene_list_file:
                for gene_info in self._gene_infos:
                    gene_list_file.write(gene_info.name + '\n')
                    if gene_info['chrom'] not in valid_staar_chromosomes:
                        valid_staar_chromosomes.add(gene_info['chrom'])
                gene_list_file.close()

        # 1. Run the STAAR NULL model
        print("Running STAAR Null Model(s)...")
        thread_utility = ThreadUtility(self._association_pack.threads,
                                       error_message='A STAAR thread failed',
                                       incrementor=10,
                                       thread_factor=1)
        for phenoname in self._association_pack.pheno_names:
            thread_utility.launch_job(staar_null,
                                      phenoname=phenoname,
                                      is_binary=self._association_pack.is_binary,
                                      found_quantitative_covariates=self._association_pack.found_quantitative_covariates,
                                      found_categorical_covariates=self._association_pack.found_categorical_covariates)
        thread_utility.collect_futures()

        # 2. Run the actual per-gene association tests
        print("Running STAAR masks * chromosomes...")
        thread_utility = ThreadUtility(self._association_pack.threads,
                                       error_message='A STAAR thread failed',
                                       incrementor=10,
                                       thread_factor=1)

        for phenoname in self._association_pack.pheno_names:
            for tarball_prefix in self._association_pack.tarball_prefixes:
                for chromosome in valid_staar_chromosomes:
                    if Path(tarball_prefix + "." + chromosome + ".STAAR.matrix.rds").exists():
                        thread_utility.launch_job(staar_genes,
                                                  tarball_prefix=tarball_prefix,
                                                  chromosome=chromosome,
                                                  phenoname=phenoname,
                                                  has_gene_info=True)

        # 3. Print a preliminary STAAR output
        print("Finalising STAAR outputs...")
        completed_staar_files = []
        # And gather the resulting futures
        for result in thread_utility:
            tarball_prefix, finished_chromosome, phenoname = result
            completed_staar_files.append(
                f'{tarball_prefix}.{phenoname}.{finished_chromosome}.STAAR_results.tsv')

        # 4. Annotate STAAR output
        process_staar_outputs(completed_staar_files, self._output_prefix,
                              self._association_pack.is_snp_tar, self._association_pack.is_gene_tar, self._gene_infos)
