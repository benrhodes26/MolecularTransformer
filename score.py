#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse
import os
from rdkit import Chem
import pandas as pd
from utils import ds2sm, sf2sm


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''


def get_rank(row, base, max_rank):
    for i in range(1, max_rank+1):
        if row['target'] == row['{}{}'.format(base, i)]:
            return i
    return 0


def main(opt):

    strip_and_split = lambda x: ''.join(x.strip().split(' '))
    with open(opt.targets, 'r') as f:

        if opt.mol_format == "smiles":
            targets = [strip_and_split(line) for line in f.readlines()]

        elif opt.mol_format == "deepsmiles":
            convert_func = lambda x: x if opt.no_canonicalize else lambda x: canonicalize_smiles(ds2sm(x))
            targets = [convert_func(strip_and_split(line)) for line in f.readlines()]

        elif opt.mol_format == "selfies":
            convert_func = lambda x: x if opt.no_canonicalize else lambda x: canonicalize_smiles(sf2sm(x))
            targets = [convert_func(strip_and_split(line)) for line in f.readlines()]

        else:
            raise ValueError("Unrecognised molecular format: {}. "
                             "Expected 'smiles' or 'deepsmiles' or 'selfies'.".format(opt.mol_format))

    predictions = [[] for _ in range(opt.beam_size)]

    output_df = pd.DataFrame(targets)
    output_df.columns = ['target']
    total = len(output_df)

    with open(opt.predictions, 'r') as f:

        # for non-smiles outputs, if conversion fails ignore (just predict empty string)
        if opt.mol_format == "smiles":
            for i, line in enumerate(f.readlines()):
                pred_smile = ''.join(line.strip().split(' '))
                predictions[i % opt.beam_size].append(pred_smile)

        elif opt.mol_format == "deepsmiles":
            for i, line in enumerate(f.readlines()):
                try:
                    convert_func = lambda x: x if opt.no_canonicalize else lambda x: ds2sm(x)
                    pred_smile = convert_func(strip_and_split(line))
                except:
                    pred_smile = ""
                predictions[i % opt.beam_size].append(pred_smile)

        elif opt.mol_format == "selfies":
            for i, line in enumerate(f.readlines()):
                try:
                    convert_func = lambda x: x if opt.no_canonicalize else lambda x: sf2sm(x)
                    pred_smile = convert_func(strip_and_split(line))
                except:
                    pred_smile = ""
                predictions[i % opt.beam_size].append(pred_smile)

    for i, preds in enumerate(predictions):
        output_df['prediction_{}'.format(i + 1)] = preds

        if opt.no_canonicalize:
            output_df['canonical_prediction_{}'.format(i + 1)] = output_df['prediction_{}'.format(i + 1)]  # do nothing
        else:
            output_df['canonical_prediction_{}'.format(i + 1)] = \
                output_df['prediction_{}'.format(i + 1)].apply(lambda x: canonicalize_smiles(x))

    output_df['rank'] = output_df.apply(lambda row: get_rank(row, 'canonical_prediction_', opt.beam_size), axis=1)
    correct = 0

    os.makedirs(opt.outdir, exist_ok=True)

    fname = "invalid_smiles_percent.txt"
    if opt.no_canonicalize: fname = "no_canonicalize_" + fname
    with open(opt.outdir + fname, 'a') as f:
        for i in range(1, opt.beam_size+1):
            correct += (output_df['rank'] == i).sum()
            invalid_smiles = (output_df['canonical_prediction_{}'.format(i)] == '').sum()

            result_str = 'Top-{}: {:.1f}% || Invalid SMILES {:.2f}%\n'.format(i, correct/total*100, invalid_smiles/total*100)
            f.write(result_str)
            print(result_str)

    fname = "predictions_targets"
    if opt.no_canonicalize: fname = "no_canonicalize_" + fname
    output_df.to_csv(opt.outdir + fname + ".csv")
    output_df.to_string(opt.outdir + fname + ".txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='score_predictions.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-beam_size', type=int, default=5,
                       help='Beam size')
    parser.add_argument('-invalid_smiles', action="store_true",
                       help='Show % of invalid SMILES')
    parser.add_argument('-no_canonicalize', action="store_true",
                       help='Do not convert everything to canonicalized smiles format before computing accuracy')
    parser.add_argument('-predictions', type=str, default="",
                       help="Path to file containing the predictions")
    parser.add_argument('-targets', type=str, default="",
                       help="Path to file containing targets")
    parser.add_argument('-mol_format', type=str, default="smiles",
                        help="smiles/deepsmiles or selfies")
    parser.add_argument('-outdir', type=str, default="experiments/",
                        help="output directory")

    opt = parser.parse_args()
    main(opt)
