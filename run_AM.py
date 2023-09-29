import argparse


parser = argparse.ArgumentParser()
parser.add_argument("sequenceFile", help="Sequence to be predicted in fasta format", type = str)
parser.add_argument("proteinID", help="Name of target protein", type = str)
parser.add_argument("referenceAA", help="Reference AA", type = str)
parser.add_argument("position", help="AA position", type = int)
parser.add_argument("targetAA", help="Target AA", type = str)
parser.add_argument("-db","--dataBaseDIR", help="DIR to alphafoldDB", type = str, default="/mnt/f/alphafoldDB")
parser.add_argument("-o","--outPutDIR", help="Output DIR", type = str)

args = parser.parse_args()

from alphamissense.data import pipeline_missense
from alphamissense.model import config
from alphamissense.model import modules_missense
import jax
import haiku as hk

protein_sequence_file = args.sequenceFile
DATABASES_DIR = args.dataBaseDIR


pipeline = pipeline_missense.DataPipeline(
    jackhmmer_binary_path="/usr/bin/jackhmmer",  # Typically '/usr/bin/jackhmmer'.
    protein_sequence_file=protein_sequence_file,
    uniref90_database_path=DATABASES_DIR + '/uniref90/uniref90.fasta',
    mgnify_database_path=DATABASES_DIR + '/mgnify/mgy_clusters_2022_05.fa',
    small_bfd_database_path=DATABASES_DIR + '/small_bfd/bfd-first_non_consensus_sequences.fasta',
)

sample = pipeline.process(
    protein_id=args.proteinID,  # Sequence identifier in the FASTA file.
    reference_aa=args.referenceAA,  # Single capital letter, e.g. 'A'.
    alternate_aa=args.targetAA,
    position=args.position,  # Integer, note that the position is 1-based!
    msa_output_dir=args.outPutDIR,
)

def _forward_fn(batch):
    model = modules_missense.AlphaMissense(config.model_config().model)
    return model(batch, is_training=False, return_representations=False)

random_seed = 0
prng = jax.random.PRNGKey(random_seed)
params = hk.transform(_forward_fn).init(prng, sample)
apply = jax.jit(hk.transform(_forward_fn).apply)
output = apply(params, prng, sample)
print(output['logit_diff']['variant_pathogenicity'])