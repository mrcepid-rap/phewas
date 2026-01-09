from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import dxpy
from general_utilities.import_utils.file_handlers.input_file_handler import InputFileHandler
from general_utilities.import_utils.import_lib import BGENInformation, TarballType

from general_utilities.import_utils.module_loader.association_pack import AssociationPack, ProgramArgs


@dataclass
class PhewasProgramArgs(ProgramArgs):
    association_tarballs: InputFileHandler
    gene_ids: List[str]
    sparse_grm: InputFileHandler
    sparse_grm_sample: InputFileHandler
    bgen_index: InputFileHandler


    def __post_init__(self):
        """@dataclass automatically calls this method after calling its own __init__().

        This is required in the subclass because dataclasses do not call the __init__ of their super o.0

        """
        self._check_opts()

    def _check_opts(self):
        pass


class PhewasAssociationPack(AssociationPack):

    def __init__(self, association_pack: AssociationPack,
                 tarball_prefixes: List[Path], gene_ids: List[str], sparse_grm: Path,
                 sparse_grm_sample: Path, bgen_dict: Dict[str, BGENInformation], tarball_type: TarballType):

        super().__init__(association_pack.is_binary, association_pack.sex, association_pack.threads,
                         association_pack.pheno_names,
                         association_pack.found_quantitative_covariates, association_pack.found_categorical_covariates,
                         association_pack.cmd_executor, association_pack.final_covariates, association_pack.inclusion_samples,
                         association_pack.exclusion_samples, association_pack.transcript_index)

        self.tarball_type = TarballType
        self.tarball_prefixes = tarball_prefixes
        self.gene_ids = gene_ids
        self.sparse_grm = sparse_grm
        self.sparse_grm_sample = sparse_grm_sample
        self.bgen_dict = bgen_dict
        self.tarball_type = tarball_type

