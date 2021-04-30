import os

import torch
import torch.nn as nn

from speech2image.encoder import Encoder

import speechbrain
from speechbrain.pretrained import EncoderDecoderASR


class CRDNNEncoder(Encoder):
    def __init__(self, asr_model, **kwargs):
        super().__init__()

        output_dim = kwargs.get('output_dim', 512)

        self.asr_model = asr_model

        #TODO: get input dim from hparams
        self.seq = nn.Sequential(
                    nn.Linear(512, output_dim)
                )


    def forward(self, speech, speech_lens, sample_rate=16000):
        if self.asr_model is None:
            raise ValueError("cannot encode if asr_model is None")

        assert speech.size(0) == speech_lens.size(0)
        assert len(speech.size()) in (2, 3)
        assert torch.max(speech_lens) <= 1

        #normalized = self.normalize(speech, sample_rate)
        
        encoded = self.asr_model.encode_batch(speech, speech_lens)
        
        # TODO: come up with a better temporal reduction
        encoded, _ = torch.max(encoded, dim=1)

        return self.seq(encoded)


    def normalize(self, speech_wav, sample_rate=16000):
        return self.asr_model.normalizer(speech_wav, sample_rate)


    @staticmethod
    def from_pretrained(model_str="speechbrain/asr-crdnn-rnnlm-librispeech", save_dir="./.model_cache", **kwargs):
        os.makedirs(save_dir, exist_ok=True)

        asr_model = EncoderDecoderASR.from_hparams(
                source=model_str,
                savedir=save_dir
            )

        model = CRDNNEncoder(asr_model, kwargs)

        return model

