# Model Card — Alzheimer's Early Detection AI

## Model

ResNet50 CNN with multimodal fusion.

## Inputs

- MRI axial slice
- MMSE
- nWBV
- Age
- EDUC

## Outputs

- CN: Cognitively Normal
- MCI: Mild Cognitive Impairment
- MA: Alzheimer's / Demented class

## Intended Use

Research and educational use only. Not for medical diagnosis.

## Limitations

- Small dataset
- OASIS-2 demographic limitations
- No clinical validation
- Model predictions require expert interpretation

## Dataset Citation

OASIS-2 / Marcus et al. (2010), ODC-by licence.
