# espnet-based encoder

import torch
import numpy as np

from espnet_model_zoo.downloader import ModelDownloader
from espnet2.torch_utils.device_funcs import to_device
from espnet2.tasks.asr import ASRTask

class ESPnetEncoder(torch.nn.Module):
    def __init__(self, asr_model=None, dtype=torch.float32, device="cuda"):
        super().__init__()
        
        self.asr_model = asr_model
        self.dtype = dtype
        self.device = torch.device(device)

    def forward(self, speech):
        # speech: torch.tensor, np.ndarray

        if self.asr_model is None:
            raise ValueError("ESPnet asr model not initialized")

        # Input as audio signal
        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)

        # data: (Nsamples,) -> (1, Nsamples)
        # speech = speech.unsqueeze(0).to(self.dtype)
        speech = speech.to(self.dtype)
        
        lengths = speech.new_full([speech.shape[0]], dtype=torch.long, fill_value=speech.size(1))
        batch = {"speech": speech, "speech_lengths": lengths}
        
        # to device
        batch = to_device(batch, device=self.device)

        # forward
        enc, _ = self.asr_model.encode(**batch)
        
        return enc

    @staticmethod
    def from_pretrained(model_name="Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best",
                            device="cuda",
                            dtype="float32"):
        d = ModelDownloader()

        model_params = d.download_and_unpack(model_name)
        
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            device=device,
            config_file=model_params["asr_train_config"],
            model_file=model_params["asr_model_file"]        
        )

        dtype = getattr(torch, dtype)
        asr_model.to(dtype).eval()

        return ESPnetEncoder(asr_model, dtype=dtype)
