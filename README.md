# Welcome to the Hackathon Template!

This time we have a use case for you - your data lies in `data/` folder.
Improve BTE, glue the ecosystem together, reach the north star

Communities: A list of communities that are part of the ecosystem.
Events: A list of events that are part of the ecosystem.
Communities to Events: Which event is organized by which community?


## Getting Started

```bash
# install uv, more about uv in `.docs/uv_crash_course.md`
curl -LsSf https://astral.sh/uv/install.sh | sh

# install the dependencies
uv sync
```
Create a `.env` file in the root of the repository and add your API keys.
Look at the `.env.example` file for reference.

This repository is a starter template with guides for:
- Using Gemini
- Using OpenRouter
- Downloading a dataset with HuggingFace
- A crash course on `uv` (Python package and environment manager)

## Examples

-   [`gemini_example.py`](examples/gemini_example.py): Demonstrates basic usage of the Google Gemini API.
-   [`openrouter_example.py`](examples/openrouter_example.py): Shows how to interact with models via the OpenRouter API.
-   [`load_dataset.py`](examples/load_dataset.py): Shows how to download a dataset from HuggingFace.

## Documentation

-   [`.docs/uv_crash_course.md`](.docs/uv_crash_course.md): A quick guide to using the `uv` Python package manager.







