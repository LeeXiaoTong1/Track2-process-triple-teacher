#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, os

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--score_csv', required=True)
    ap.add_argument('--threshold', type=float, required=True)
    ap.add_argument('--out_csv', required=True)
    args=ap.parse_args()
    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    n=0
    with open(args.score_csv,'r',encoding='utf-8-sig') as fin, open(args.out_csv,'w',encoding='utf-8',newline='') as fout:
        r=csv.DictReader(fin)
        w=csv.writer(fout)
        w.writerow(['name','predict'])
        for row in r:
            score=float(row['score'])
            pred='real' if score >= args.threshold else 'fake'
            w.writerow([row['name'].strip(), pred])
            n+=1
    print('Saved:', args.out_csv, 'n=', n, 'threshold=', args.threshold)
if __name__=='__main__':
    main()
