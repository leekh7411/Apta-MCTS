import os
import argparse
import pandas as pd
from mcts import AptaMCTS
from utils import read_fa_file
from collections import defaultdict


def candidates_to_csv(candidates, path):
    """

    Args:
        candidates (list): Sampled top-k candidate results
        path: Result file path

    Returns:

    """
    output_dict = defaultdict(lambda: [])
    for score, sequence, secondary_structure, mfe in candidates:
        output_dict['aptamer_protein_interaction_score'].append(score)
        output_dict['primary_sequence'].append(sequence)
        output_dict['secondary_structure'].append(secondary_structure)
        output_dict['minimum_free_energy'].append(mfe)

    output_df = pd.DataFrame.from_dict(output_dict)
    output_df.to_csv(path, index=False)
    print('Result .csv file saved in path: {}\n'.format(path))


def aptamer_monte_carlo_tree_search(inp_protein_fa,
                                    ex_protein_fa,
                                    top_k,
                                    bp,
                                    num_iterations,
                                    score_function_path,
                                    forward_sequence,
                                    backward_sequence,
                                    output_dir):
    """ Generate Candidate Aptamer Sequences using Monte-Carlo Tree Search
        based on the API-classifier scores

    Args:
        inp_protein_fa (str): Input target protein sequences .fasta format file to be interacted
        ex_protein_fa (str): Exclusive protein sequences .fasta format file
        top_k (int): Number of candidate aptamer results for each output .csv file
        bp (int): Length of candidate aptamer sequence except forward and backward subsequences
        num_iterations (int): Number of iterations for each Monte-Carlo Tree Search step
        score_function_path (str): Pre-trained Aptamer-Protein Interaction classifier path
        forward_sequence (str): Forward subsequence for candidate sequences
        backward_sequence (str): Backward subsequence for candidate sequences
        output_dir (str): Directory to save sampled candidate results

    Returns:
        None
    """
    generator = AptaMCTS(score_function_path)
    protein_names, protein_sequences = read_fa_file(inp_protein_fa)
    if ex_protein_fa is not None:
        ex_protein_names, ex_protein_sequences = read_fa_file(ex_protein_fa)
        ex_proteins = (ex_protein_names, ex_protein_sequences)
    else:
        ex_proteins = ([], [])

    for p_name, p_seq in zip(protein_names, protein_sequences):
        candidate_aptamers = generator.sampling(p_seq,
                                                bp,
                                                top_k,
                                                num_iterations,
                                                ex_proteins,
                                                forward_sequence,
                                                backward_sequence)

        output_csv_path = os.path.join(output_dir, '{}.csv'.format(p_name))
        candidates_to_csv(candidate_aptamers, path=output_csv_path)


def main(args):
    inp_protein_file_path = args.input_protein
    ex_protein_file_path = args.ex_protein
    top_k = int(args.top_k)
    candidate_aptamer_bp = int(args.bp_size)
    num_iterations = int(args.num_iterations)
    score_function_path = args.score_function
    output_dir = args.output_dir
    fwd_seq = args.forward_sequence
    bwd_seq = args.backward_sequence
    aptamer_monte_carlo_tree_search(inp_protein_fa=inp_protein_file_path,
                                    ex_protein_fa=ex_protein_file_path,
                                    top_k=top_k,
                                    bp=candidate_aptamer_bp,
                                    num_iterations=num_iterations,
                                    score_function_path=score_function_path,
                                    forward_sequence=fwd_seq,
                                    backward_sequence=bwd_seq,
                                    output_dir=output_dir)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-protein', type=str, required=True,
                        help='Target protein sequences (.fa or .fasta) file path')
    parser.add_argument('-k', '--top-k', type=int, default=5,
                        help='Number of candidate aptamer sequences to be selected for each iteration')
    parser.add_argument('-bp', '--bp-size', type=int, default=30,
                        help='Length of candidate aptamer sequences')
    parser.add_argument('-n', '--num-iterations', type=int, default=100,
                        help='Number of iterations for each MCTS process')
    parser.add_argument('-s', '--score-function', type=str, required=True,
                        help='Score function (pre-trained Aptamer-Protein Interaction Classifier) '
                             'for candidate aptamer samples')
    parser.add_argument('-e', '--ex-protein', type=str, required=False, default=None,
                        help='The protein sequences (.fa or .fasta) file path do not want to interact with '
                             'candidate aptamers in sampling process')
    parser.add_argument('-o', '--output-dir', type=str, required=False, default='examples/',
                        help='Directory to save result .csv file')
    parser.add_argument('-fwd', '--forward-sequence', type=str, required=False, default=None,
                        help='A sequence that concatenated in forward of candidate sequence')
    parser.add_argument('-bwd', '--backward-sequence', type=str, required=False, default=None,
                        help='A sequence that concatenated in backward of candidate sequence')
    args = parser.parse_args()
    main(args)
