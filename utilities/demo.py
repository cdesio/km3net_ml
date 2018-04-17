#!/usr/bin/python
import argparse

parser = argparse.ArgumentParser(description='This is a demo script')

parser.add_argument('-mu', '--numu', help='numu file name', required=True)
parser.add_argument('-e', '--nue', help='nue file name', required=True)
parser.add_argument('--xnumu', help = 'out numu file', required=True)
parser.add_argument('--xnue', help = 'out nue file', required=True)
args = parser.parse_args()
print(args.numu, args.nue, args.xnumu, args.xnue)


print("Input file: %s" % args.numu)
print("Output file: %s"% args.xnumu)
