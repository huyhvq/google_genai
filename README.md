# Google Gemini Pipe

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-%3E=3.7-blue)](https://www.python.org/downloads/)

**OpenWebUI Pipe for Google Gemini with API Key Rotation**

This project provides a Python pipe designed to seamlessly integrate Google's powerful Gemini models into [OpenWebUI](https://github.com/open-webui/open-webui) or similar platforms. A key feature is its support for **API key rotation**, allowing you to use multiple Google API keys for enhanced reliability, rate limit management, and key security.

**Author:** [huyhvq](https://github.com/huyhvq)

**Repository URL:** [https://github.com/huyhvq/google_genai](https://github.com/huyhvq/google_genai)

## Features

- **Google Gemini Integration:**  Leverages the `google-generativeai` Python SDK to interact with Google's Gemini family of models (Gemini Pro, Gemini Pro Vision, etc.).
- **API Key Rotation:** Supports providing a comma-separated list of Google API keys via environment variables. The pipe automatically rotates through these keys for each API call, helping to avoid rate limits and improve service availability.
- **OpenWebUI Compatibility:** Designed to function as a pipe within the OpenWebUI framework, easily adding Google Gemini models to your OpenWebUI instance.
- **Model Listing:** Automatically retrieves and registers available Google Gemini models that support content generation, making them selectable in OpenWebUI.
- **Safety Settings:** Includes an option to use permissive safety settings (BLOCK_NONE) via an environment variable, allowing for more flexible content generation (use with caution!).
- **Detailed Logging (Debug Mode):**  Provides comprehensive logging for debugging and monitoring, which can be enabled or disabled via a `DEBUG` flag in the code.
- **System Message Handling:**  Correctly processes system messages within conversations to guide the Gemini model's behavior.
- **Multi-part Content Support:** Handles text and image content within messages, enabling multimodal interactions with Gemini models that support them (like Gemini Pro Vision).
- **Streaming Support:** (To be implemented/verified - check code comments) Potentially supports streaming responses from Gemini for a more interactive user experience.

## Installation

This pipe is intended to be used within an OpenWebUI environment.  You will typically place this code within the `pipes` directory of your OpenWebUI installation.

1. **Clone or Download:** Download the `google_genai` directory (or the Python file if it's a single file pipe) into the `pipes` directory of your OpenWebUI instance.
2. **Dependencies:** Ensure you have the `google-generativeai` Python library installed. You can install it using pip:

   ```bash
   pip install google-generativeai
   ```

## Configuration

### Environment Variables

The Google Gemini Pipe relies on environment variables for configuration:

- **`GOOGLE_API_KEY`:**  **(Required)**  Set this to your Google API key. To use API key rotation, provide a comma-separated list of API keys. For example:

  ```bash
  export GOOGLE_API_KEY="YOUR_API_KEY_1,YOUR_API_KEY_2,YOUR_API_KEY_3"
  ```

  If you only have one API key, just set it directly:

  ```bash
  export GOOGLE_API_KEY="YOUR_SINGLE_API_KEY"
  ```

- **`USE_PERMISSIVE_SAFETY`:** (Optional) Set this to `"true"`, `"1"`, or `"yes"` to enable permissive safety settings (BLOCK_NONE for all safety categories). This disables Google's safety filters and allows potentially harmful content to be generated. **Use with extreme caution and only if you understand the risks.**  If not set or set to any other value, default Google safety settings will be applied.

  ```bash
  export USE_PERMISSIVE_SAFETY="true"
  ```

### Valves (OpenWebUI Configuration)

Within OpenWebUI's settings, you will find "Valves" for each pipe. For the "Google Gemini Pipe", you should configure the `GOOGLE_API_KEY` valve. While the environment variable is the primary way to set the API key, the Valves in OpenWebUI can provide another layer of configuration within the UI itself.  Setting the `GOOGLE_API_KEY` environment variable is generally recommended for persistent configuration.

## Usage in OpenWebUI

1. **Start or Restart OpenWebUI:** Ensure your OpenWebUI instance is running or restart it after placing the pipe code in the `pipes` directory and setting the environment variables.
2. **Select Model:** In the OpenWebUI chat interface, you should now see Google Gemini models listed in the model selection dropdown. The model names will correspond to the display names provided by the Google Gemini API (e.g., "Gemini Pro", "Gemini Pro Vision").
3. **Start Chatting:** Select a Google Gemini model and begin your conversation. The pipe will handle communication with the Google Gemini API, API key rotation, and response processing.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details (if you have a LICENSE file, otherwise, you might want to create one).

## Disclaimer

This project is provided as-is, without warranty. Use it at your own risk. Be mindful of Google Gemini API usage and billing. Permissive safety settings should be used with caution and only when appropriate for your use case.

**Enjoy using Google Gemini with API Key Rotation in OpenWebUI!**
```
