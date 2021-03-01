#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import argparse
from rdkit import Chem
import pandas as pd
from utils import ds2sm,sf2sm

def main(opt):
    total_line = 0
    accuracies = [0]*opt.beam_size
    tgtfile = open(opt.targets, "r")
    predfile = open(opt.predictions, "r")
    for tl in tgtfile: 
        tltokens = tl.split(" ")
        for i in range(opt.beam_size):
            pl = predfile.readline()
            prtokens = pl.split(" ")
            if all([t==p for t,p in zip(tltokens,prtokens)]):
                accuracies[i]+=1
        total_line += 1
    accum = 0
    for i,acc in enumerate(accuracies):
        accum+=acc/total_line
        print("Top {}: ".format(i),  accum*100, "%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='score_predictions.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-beam_size', type=int, default=5,
                       help='Beam size')
    parser.add_argument('-invalid_smiles', action="store_true",
                       help='Show % of invalid SMILES')
    parser.add_argument('-predictions', type=str, default="",
                       help="Path to file containing the predictions")
    parser.add_argument('-targets', type=str, default="",
                       help="Path to file containing targets")
    parser.add_argument('-mol_format', type=str, default="smiles",
                        help="smiles/deepsmiles or selfies")

    opt = parser.parse_args()
    main(opt)
