import dxpy
from general_utilities.import_utils.import_lib import ingest_tarballs, ingest_wes_bgen, TarballType
from general_utilities.import_utils.module_loader.ingest_data import IngestData

from phewas.phewas_association_pack import PhewasProgramArgs, PhewasAssociationPack


class PhewasIngestData(IngestData):

    def __init__(self, parsed_options: PhewasProgramArgs):
        super().__init__(parsed_options)

        # Put additional options/covariate processing required by this specific package here
        tarball_type, tarball_prefixes = ingest_tarballs(parsed_options.association_tarballs)
        bgen_dict = ingest_wes_bgen(parsed_options.bgen_index)

        sparse_grm = parsed_options.sparse_grm.get_file_handle()
        sparse_grm_sample = parsed_options.sparse_grm_sample.get_file_handle()

        if tarball_type not in (TarballType.SNP, TarballType.GENE) and parsed_options.gene_ids is None:
            raise dxpy.AppError('Must provide gene IDs when NOT using a SNP/GENE tarball!')

        # Put additional covariate processing specific to this module here
        self.set_association_pack(PhewasAssociationPack(self.get_association_pack(),
                                                        tarball_prefixes=tarball_prefixes,
                                                        gene_ids=parsed_options.gene_ids,
                                                        sparse_grm=sparse_grm,
                                                        sparse_grm_sample=sparse_grm_sample,
                                                        bgen_dict=bgen_dict,
                                                        tarball_type=tarball_type))
