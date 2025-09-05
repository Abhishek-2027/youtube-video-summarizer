import logging
from typing import Dict, List, Optional
from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer
from haystack.modeling.utils import get_device

logger = logging.getLogger(__name__)

class LlamaCPPInvocationLayer(PromptModelInvocationLayer):
    """
    A custom PromptModelInvocationLayer to interface with models loaded via llama-cpp-python.
    This class is required to use local Llama models with Haystack's PromptNode.
    """

    def __init__(self, model_name_or_path: str, use_gpu: bool = False, **kwargs):
        """
        Initializes the LlamaCPPInvocationLayer.
        :param model_name_or_path: Path to the GGUF model file (e.g., 'llama-2-7b-32k-instruct.Q4_K_S.gguf').
        :param use_gpu: Whether to use GPU acceleration. Requires `llama-cpp-python` to be built with CUDA support.
        :param kwargs: Additional keyword arguments for the Llama model, such as `max_length`.
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "LlamaCPPInvocationLayer requires the 'llama-cpp-python' library. "
                "Please install it with `pip install llama-cpp-python`."
            )

        super().__init__(model_name_or_path, **kwargs)

        # Determine the number of GPU layers based on the `use_gpu` flag
        n_gpu_layers = -1 if use_gpu and get_device() == "cuda" else 0
        if n_gpu_layers == -1:
            logger.info("Using GPU acceleration for LlamaCPP model.")
        else:
            logger.info("Not using GPU acceleration for LlamaCPP model. n_gpu_layers set to 0.")

        self.model = Llama(
            model_path=model_name_or_path,
            n_gpu_layers=n_gpu_layers,
            verbose=False, # Set to True for verbose logging
            **kwargs
        )
        self.max_length = kwargs.get("max_length", 512)
        logger.info(f"Initialized Llama model with max_length: {self.max_length}")

    def invoke(self, prompt: str, **kwargs) -> List[str]:
        """
        Generates a completion for the given prompt using the Llama model.
        :param prompt: The input prompt string.
        :param kwargs: Additional generation parameters.
        :return: A list containing the generated text completion.
        """
        try:
            # The streaming interface returns a generator, so we need to iterate over it
            completion = self.model.create_completion(
                prompt=prompt,
                max_tokens=kwargs.get("max_length", self.max_length),
                stream=False, # Streaming can be set to True for real-time output
            )
            generated_text = completion['choices'][0]['text']
            return [generated_text]
        except Exception as e:
            logger.error(f"Failed to generate text from prompt: {e}")
            return [""]

    def _ensure_text_is_list(self, text):
        """
        Ensures the output is a list of strings, as expected by Haystack.
        """
        if isinstance(text, str):
            return [text]
        return text

    def get_token_ids(self, text: str) -> List[int]:
        """
        Converts a string of text into a list of token IDs.
        """
        return self.model.tokenize(text.encode("utf-8"))

    def get_token_count(self, text: str) -> int:
        """
        Gets the number of tokens in a string.
        """
        return len(self.get_token_ids(text))

    def get_max_tokens(self) -> int:
        """
        Returns the maximum number of tokens supported by the model.
        """
        # The maximum context length depends on the model architecture and is often
        # a configuration parameter. For now, we return the specified max_length.
        return self.max_length
