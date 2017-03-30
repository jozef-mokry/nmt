#! /bin/bash
# Compute the blue scores of the models given on STDIN
if [ ! $# -eq 2 ]; then
  echo "Usage: $0 ref.tok output.bpe" 1>&2;
  exit 1;
fi

SCRIPTS="./"
postprocess=$SCRIPTS"postprocess.sh"
multibleu=$SCRIPTS"multi-bleu"
ref=$1
out=$2

$postprocess < $out > $out.postprocessed

## get BLEU
BLEU=`$multibleu $ref $out.postprocessed | cut -f 3 -d ' ' | cut -f 1 -d ','`
echo "BLEU: $BLEU";
