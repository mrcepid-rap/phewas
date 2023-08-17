import dxpy

from general_utilities.import_utils.genetics_loader import GeneticsLoader
from general_utilities.import_utils.import_lib import ingest_tarballs
from general_utilities.import_utils.module_loader.ingest_data import IngestData
from phewas.phewas_association_pack import PhewasProgramArgs, PhewasAssociationPack


class PhewasIngestData(IngestData):

    def __init__(self, parsed_options: PhewasProgramArgs):
        super().__init__(parsed_options)

        # Put additional options/covariate processing required by this specific package here
        is_snp_tar, is_gene_tar, named_prefix, tarball_prefixes = ingest_tarballs(parsed_options.association_tarballs)

        GeneticsLoader.ingest_sparse_matrix(parsed_options.sparse_grm,
                                            parsed_options.sparse_grm_sample)

        if is_snp_tar is False and is_gene_tar is False and parsed_options.gene_ids is None:
            raise dxpy.AppError('Must provide gene IDs when NOT using a SNP/GENE tarball!')

        # Put additional covariate processing specific to this module here
        self.set_association_pack(PhewasAssociationPack(self.get_association_pack(),
                                                        is_snp_tar, is_gene_tar, tarball_prefixes,
                                                        parsed_options.gene_ids))
