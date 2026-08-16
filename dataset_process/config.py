"""dataset_process/config.py"""
# config.py
from pathlib import Path


NUM_HOPS = 4
ENCODING_DIM = 16

class Config:

    DATA_ROOT = Path("Dataset")
    PDB_DIR = DATA_ROOT / "S2648"
    PSSM_FILE = DATA_ROOT / "pssm_s2648.pkl"
    CONSERVATION_DIR = DATA_ROOT / "cons_s2648"
    OUTPUT_DIR = Path("./pkl")


    CONTACT_CUTOFF = 8.0  # Å


    NODE_FEATURE_DIM = 53
    EDGE_FEATURE_DIM = 16


    DDG_FILE = DATA_ROOT / "S2648.csv"


Config.OUTPUT_DIR.mkdir(exist_ok=True)


# config/feature_config.py


class FeatureConfig:
    'Implementation of FeatureConfig.'

    ONE_HOT_DIM = 20
    SEQ_PROPS_DIM = 12

    STRUCTURE_DIM = 14

    EVOLUTION_DIM = 21


    DISTANCE_FEATURES_DIM = 4
    SEQUENCE_FEATURES_DIM = 4
    CONTACT_FEATURES_DIM = 4
    RESIDUE_FEATURES_DIM = 4

    @classmethod
    def get_total_node_dim(cls):
        'Get total node dim.'
        return cls.ONE_HOT_DIM + cls.SEQ_PROPS_DIM + cls.STRUCTURE_DIM + cls.EVOLUTION_DIM

    @classmethod
    def get_total_edge_dim(cls):
        'Get total edge dim.'
        return (cls.DISTANCE_FEATURES_DIM + cls.SEQUENCE_FEATURES_DIM +
                cls.CONTACT_FEATURES_DIM + cls.RESIDUE_FEATURES_DIM)

    @classmethod
    def get_node_feature_slices(cls):
        'Get node feature slices.'
        start = 0
        slices = {}


        slices['one_hot'] = slice(start, start + cls.ONE_HOT_DIM)
        start += cls.ONE_HOT_DIM


        slices['seq_properties'] = slice(start, start + cls.SEQ_PROPS_DIM)
        start += cls.SEQ_PROPS_DIM



        # slices['structure'] = slice(start, start + cls.STRUCTURE_DIM)
        # start += cls.STRUCTURE_DIM


        slices['evolution'] = slice(start, start + cls.EVOLUTION_DIM)

        return slices

    @classmethod
    def get_edge_feature_slices(cls):
        'Get edge feature slices.'
        start = 0
        slices = {}


        slices['distance'] = slice(start, start + cls.DISTANCE_FEATURES_DIM)
        start += cls.DISTANCE_FEATURES_DIM


        slices['sequence'] = slice(start, start + cls.SEQUENCE_FEATURES_DIM)
        start += cls.SEQUENCE_FEATURES_DIM


        slices['contact'] = slice(start, start + cls.CONTACT_FEATURES_DIM)
        start += cls.CONTACT_FEATURES_DIM


        slices['residue'] = slice(start, start + cls.RESIDUE_FEATURES_DIM)

        return slices

    @classmethod
    def verify_dimensions(cls, features):
        'Verify dimensions.'
        expected_dim = cls.get_total_node_dim()
        actual_dim = features.shape[-1]
        if actual_dim != expected_dim:
            raise ValueError(f"Feature dimension mismatch! Expected {expected_dim} but got {actual_dim}")
        return True