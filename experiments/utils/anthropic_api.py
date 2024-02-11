import config
from anthropic import Anthropic

import os


# Set OpenAPI key from environment or config file
api_key = os.environ.get("ANTHROPIC_API_KEY")
if (api_key is None or api_key == "") and os.path.isfile(
    os.path.join(os.getcwd(), "keys.cfg")
):
    cfg = config.Config("keys.cfg")
    api_key = cfg.get("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)


def ChatAnthropic(messages, max_tokens=512, temperature=0, top_p=1, system=""):
    response = client.beta.messages.create(
        model="claude-2.1",
        messages=messages,
        system=system,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return response.content[0].text


if __name__ == "__main__":
    pass
