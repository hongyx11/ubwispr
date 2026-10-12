"""Actionable model-access errors without displaying raw Hub responses."""
from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_url
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError


class ModelAccessError(RuntimeError):
    pass


def access_error(model_id: str, error: BaseException) -> ModelAccessError | None:
    # Transformers wraps Hub exceptions in OSError; inspect the cause rather
    # than relying on wording in a server response.
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        unauthorized = isinstance(current, GatedRepoError) or (
            isinstance(current, HfHubHTTPError) and current.response.status_code in (401, 403)
        )
        if unauthorized:
            url = f"https://huggingface.co/{model_id}"
            if get_token():
                message = (
                    f"Hugging Face denied access to {model_id}. Open {url} with the account "
                    "associated with your token and accept/request model access. Ensure your "
                    "token has read access to this gated model; activate the ubwispr Conda "
                    "environment and run `hf auth login` "
                    "to replace an invalid or expired token, then restart Ubuntu Wispr."
                )
            else:
                message = (
                    f"No Hugging Face token found for {model_id}. Open {url} and accept/request "
                    "model access, then activate the ubwispr Conda environment, run "
                    "`hf auth login`, and restart "
                    "Ubuntu Wispr. Use a read token from the same account."
                )
            return ModelAccessError(message)
        current = current.__cause__ or current.__context__
    return None


def check_model_access(model_id: str) -> tuple[bool, str]:
    """HEAD request for config metadata; never downloads model files."""
    try:
        get_hf_file_metadata(hf_hub_url(model_id, "config.json"), timeout=5)
    except Exception as exc:
        setup_error = access_error(model_id, exc)
        if setup_error is not None:
            return False, str(setup_error)
        if isinstance(exc, HfHubHTTPError):
            return False, f"Model access check failed (HTTP {exc.response.status_code}); verify model_id and repository access"
        return False, "Could not verify Hugging Face model access; check your network connection (cached inference may still work)"
    return True, f"Hugging Face model access available: {model_id}"
