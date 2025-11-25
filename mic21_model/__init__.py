from transformers import AutoConfig, AutoModel
from .configuration_mic21 import MIC21SummarizerConfig
from .modeling_mic21 import MIC21SummarizerModel

AutoConfig.register("mic21_summarizer", MIC21SummarizerConfig)
AutoModel.register(MIC21SummarizerConfig, MIC21SummarizerModel)
