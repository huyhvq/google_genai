"""
title: Gemini API Rotation Pipe
id: gemini_api_rotation_pipe
author: huyhvq
author_url: https://github.com/huyhvq
repo_url: https://github.com/huyhvq/google_genai
version: 1.0.0
requirements: google-genai==1.8.0
license: MIT
"""

import os
import re
import asyncio
import threading
import json
from pydantic import BaseModel, Field
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, GenerateContentResponse
from typing import List, Union, Iterator, Callable, Awaitable

# Set to True for detailed logging
DEBUG = True


class Pipe:
    class Valves(BaseModel):
        GOOGLE_API_KEY: str = Field(default="")
        USE_PERMISSIVE_SAFETY: bool = Field(default=False)

    def __init__(self):
        try:
            self.id = "google_genai"
            self.type = "manifold"  # Indicates this pipe provides multiple models
            self.name = "Google: "
            self.valves = self.Valves(
                **{
                    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
                    "USE_PERMISSIVE_SAFETY": os.getenv(
                        "USE_PERMISSIVE_SAFETY", "False"
                    ).lower()
                    in ["true", "1", "yes"],
                }
            )
            self.mu = threading.Lock()
            self.current_index = 0
            if DEBUG:
                print("[INIT] Initialized Google GenAI Pipe")
        except Exception as e:
            if DEBUG:
                print(f"[INIT] Error during initialization: {e}")
        finally:
            if DEBUG:
                print("[INIT] Initialization complete.")

    # Helper function to emit messages (content or errors)
    async def _emit_message(self, event_emitter, content, msg_type="message"):
        """Emit a message or error event."""
        if not event_emitter:
            if DEBUG:
                print(
                    f"[_emit_message] No event emitter provided. Cannot emit {msg_type}."
                )
            return

        try:
            data = {"content": content}
            if msg_type == "error":
                # Optionally structure error messages differently if needed by frontend
                data = {"content": f"Error: {content}"}

            event = {
                "type": msg_type,
                "data": data,
            }

            if asyncio.iscoroutinefunction(event_emitter):
                await event_emitter(event)
            else:
                # Handle synchronous emitters if necessary, though unlikely in Open WebUI context
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(event_emitter, event)

            if DEBUG:
                log_content = content[:100] + "..." if len(content) > 100 else content
                print(f"[_emit_message] Emitted {msg_type}: '{log_content}'")
        except Exception as e:
            if DEBUG:
                print(f"[_emit_message] Error emitting {msg_type}: {e}")

    def strip_prefix(self, model_name: str) -> str:
        """
        Strip common prefixes like 'models/' or 'publishers/google/models/'
        to get the base model ID.
        """
        try:
            # More robustly remove known prefixes
            prefixes_to_remove = ["models/", "publishers/google/models/"]
            for prefix in prefixes_to_remove:
                if model_name.startswith(prefix):
                    stripped = model_name[len(prefix) :]
                    if DEBUG:
                        print(
                            f"[strip_prefix] Stripped '{prefix}' -> '{stripped}' from '{model_name}'"
                        )
                    return stripped
            # If no known prefix, return original (or apply generic rule if preferred)
            if DEBUG:
                print(f"[strip_prefix] No known prefix found in '{model_name}'")
            return model_name
        except Exception as e:
            if DEBUG:
                print(f"[strip_prefix] Error stripping prefix: {e}")
            return model_name  # Return original if stripping fails

    def get_next_api_key(self):
        try:
            if not self.valves.GOOGLE_API_KEY:
                if DEBUG:
                    print("[get_next_api_key] GOOGLE_API_KEY is not set.")
                return [
                    {
                        "id": "error-no-key",
                        "name": "API Key missing. Please set GOOGLE_API_KEY in Valves.",
                        "error": "API Key missing",
                    }
                ]

            with self.mu:
                api_keys = self.valves.GOOGLE_API_KEY.replace(" ", "").split(",")
                key = api_keys[self.current_index]
                self.current_index = (self.current_index + 1) % len(api_keys)

            if DEBUG:
                print(f"[get_next_api_key] Invoke API with key: {key}.")
            return key

        except Exception as e:
            if DEBUG:
                print(f"[get_next_api_key] Error get rotation API key: {e}")
            return [
                {
                    "id": "error-fetch",
                    "name": f"Could not fetch models: {str(e)}",
                    "error": str(e),
                }
            ]

    def get_google_models(self):
        """Retrieve available Google models."""
        try:
           if not self.valves.GOOGLE_API_KEY:
               if DEBUG:
                   print("[get_google_models] GOOGLE_API_KEY is not set.")
               return [
                   {
                       "id": "error-no-key",
                       "name": "API Key missing. Please set GOOGLE_API_KEY in Valves.",
                       "error": "API Key missing",
                   }
               ]

           genai.configure(api_key=self.valves.GOOGLE_API_KEY)
           models = genai.list_models()
           models_list = list(models)

           if DEBUG:
               print(
                   f"[get_google_models] Retrieved {len(models_list)} models raw from Google."
               )

           filtered_models = []
           for model in models_list:
               # Check if the model supports content generation
               if "generateContent" in model.supported_generation_methods:
                   model_id = self.strip_prefix(model.name)
                   filtered_models.append(
                       {
                           "id": model_id,
                           "name": model.display_name,
                           # Include other potentially useful info if available
                           "description": getattr(model, "description", None),
                           "context_window": getattr(model, "input_token_limit", None),
                           "output_window": getattr(model, "output_token_limit", None),
                           "temperature": getattr(
                               model, "temperature", None
                           ),  # May not be available here
                           "top_p": getattr(
                               model, "top_p", None
                           ),  # May not be available here
                           "top_k": getattr(
                               model, "top_k", None
                           ),  # May not be available here
                       }
                   )

           if not filtered_models:
               if DEBUG:
                   print(
                       "[get_google_models] No models found supporting 'generateContent'."
                   )
               return [
                   {
                       "id": "error-no-models",
                       "name": "No compatible Google models found.",
                       "error": "No compatible models",
                   }
               ]

           if DEBUG:
               print(
                   f"[get_google_models] Found {len(filtered_models)} compatible models."
               )
           return filtered_models

        except Exception as e:
           if DEBUG:
               print(f"[get_google_models] Error fetching Google models: {e}")
           return [
               {
                   "id": "error-fetch",
                   "name": f"Could not fetch models: {str(e)}",
                   "error": str(e),
               }
           ]
        finally:
           if DEBUG:
               print("[get_google_models] Completed fetching Google models.")

    def pipes(self) -> List[dict]:
        """Register all available Google models that support generateContent."""
        return self.get_google_models()

    async def pipe(
        self, body: dict, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> None:
        """Main pipe method to process incoming requests to Google Gemini."""

        async def emit_message(content):
            await self._emit_message(__event_emitter__, content, "message")

        async def emit_error(error_message):
            await self._emit_message(__event_emitter__, error_message, "error")

        try:
            if not self.valves.GOOGLE_API_KEY:
                if DEBUG:
                    print("[pipe] GOOGLE_API_KEY is not set.")
                await emit_error("GOOGLE_API_KEY is not set in Valves.")
                return

            try:

                genai.configure(api_key=self.get_next_api_key())
                if DEBUG:
                    print("[pipe] Configured Google Generative AI with API key.")
            except Exception as e:
                if DEBUG:
                    print(f"[pipe] Error configuring Google Generative AI: {e}")
                await emit_error(f"Error configuring Google Generative AI: {e}")
                return

            # Model ID might come with prefixes, strip them if necessary
            # The 'id' returned by `pipes` should already be stripped, but handle cases
            # where the full name might be passed.
            raw_model_id = body.get("model", "")
            model_id = self.strip_prefix(raw_model_id).split(".", 1)[-1]
            if "thinking" in model_id or "gemini-2.5-pro" in model_id:
                enclosure = f"""<details>
<summary>Click to expand thoughts</summary>
Gemini SDK no more provided thought.
</details>""".strip()
                message_event = {
                    "type": "message",
                    "data": {"content": enclosure},
                }
                await __event_emitter__(message_event)

            if DEBUG:
                print(f"[pipe] Using model ID: '{model_id}' (raw: '{raw_model_id}')")
            messages = body.get("messages", [])
            if DEBUG:
                print(f"[pipe] Processing {len(messages)} incoming messages.")

            # Extract system message if present
            system_message = next(
                (msg["content"] for msg in messages if msg.get("role") == "system"),
                None,
            )
            if DEBUG and system_message:
                print(f"[pipe] Extracted system message: '{system_message[:100]}...'")

            # Format messages for Google API
            contents = []
            try:
                # Combine system message with the first user message if applicable
                first_user_message_index = -1
                if system_message:
                    for i, message in enumerate(messages):
                        if message.get("role") == "user":
                            first_user_message_index = i
                            break

                for i, message in enumerate(messages):
                    role = message.get("role")
                    if role == "system":
                        continue  # Skip system message here, handled above/below

                    content = message.get("content", "")
                    parts = []

                    # Prepend system message to the first user message content
                    if system_message and i == first_user_message_index:
                        # Assuming content is text or first part is text
                        if isinstance(content, str):
                            content = f"System Instruction: {system_message}\n\nUser Request: {content}"
                        elif (
                            isinstance(content, list)
                            and content
                            and content[0].get("type") == "text"
                        ):
                            content[0][
                                "text"
                            ] = f"System Instruction: {system_message}\n\nUser Request: {content[0].get('text', '')}"
                        else:
                            # Add system message as a separate text part if first user message isn't text
                            parts.append(
                                {"text": f"System Instruction: {system_message}"}
                            )
                        if DEBUG:
                            print(
                                "[pipe] Prepended system message to first user message."
                            )
                        system_message = None  # Ensure it's only added once

                    # Process content parts (text, image)
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                parts.append({"text": item.get("text", "")})
                            elif item.get("type") == "image_url":
                                image_url = item.get("image_url", {}).get("url", "")
                                if image_url.startswith("data:image"):
                                    # Handle base64 encoded images
                                    try:
                                        mime_type = image_url.split(":")[1].split(";")[
                                            0
                                        ]
                                        image_data = image_url.split(",", 1)[1]
                                        parts.append(
                                            {
                                                "inline_data": {
                                                    "mime_type": mime_type,
                                                    "data": image_data,
                                                }
                                            }
                                        )
                                    except Exception as img_err:
                                        if DEBUG:
                                            print(
                                                f"[pipe] Error processing base64 image: {img_err}"
                                            )
                                        await emit_error(
                                            f"Error processing image data: {img_err}"
                                        )
                                        return  # Stop processing if image is invalid
                                else:
                                    # Google API doesn't directly support URLs, this part might need adjustment
                                    # depending on how Open WebUI handles image URLs (e.g., download first).
                                    # For now, we'll skip non-data URLs or raise an error.
                                    if DEBUG:
                                        print(
                                            f"[pipe] Skipping non-data image URL: {image_url[:50]}..."
                                        )
                                    # Option: await emit_error("Direct image URLs are not supported by this pipe. Please use uploaded images.")
                                    # return
                            # Add handling for other content types if needed
                    elif isinstance(content, str):
                        parts.append({"text": content})
                    else:
                        if DEBUG:
                            print(
                                f"[pipe] Skipping message with unexpected content type: {type(content)}"
                            )
                        continue  # Skip messages with unexpected content format

                    # Map roles: 'assistant' -> 'model'
                    api_role = "model" if role == "assistant" else "user"
                    if parts:  # Only add if there are valid parts
                        contents.append({"role": api_role, "parts": parts})

                # If system message wasn't prepended (e.g., no user messages), add it now.
                if system_message:
                    contents.insert(
                        0,
                        {
                            "role": "user",
                            "parts": [
                                {"text": f"System Instruction: {system_message}"}
                            ],
                        },
                    )
                    if DEBUG:
                        print("[pipe] Added system message as initial user message.")

                if not contents:
                    await emit_error("No valid content found to send to the model.")
                    return

                if DEBUG:
                    # Avoid printing large base64 strings in logs
                    log_contents = []
                    for c in contents:
                        log_parts = []
                        for p in c["parts"]:
                            if "inline_data" in p:
                                log_parts.append(
                                    {
                                        "inline_data": {
                                            "mime_type": p["inline_data"]["mime_type"],
                                            "data": "[base64 data]",
                                        }
                                    }
                                )
                            else:
                                log_parts.append(p)
                        log_contents.append({"role": c["role"], "parts": log_parts})
                    print(f"[pipe] Processed contents for API: {log_contents}")

            except Exception as e:
                if DEBUG:
                    print(f"[pipe] Error processing messages: {e}")
                await emit_error(f"Error processing messages: {e}")
                return

            # --- API Call ---
            try:
                client = genai.GenerativeModel(model_name=model_id)
                if DEBUG:
                    print(
                        f"[pipe] Initialized GenerativeModel with model ID: '{model_id}'"
                    )
            except Exception as e:
                if DEBUG:
                    print(f"[pipe] Error initializing GenerativeModel: {e}")
                await emit_error(
                    f"Error initializing GenerativeModel for '{model_id}': {e}"
                )
                return

            # Configure generation settings
            generation_config = GenerationConfig(
                temperature=body.get("temperature", 0.7),  # Default temperature
                top_p=body.get("top_p"),  # Let Google use its default if not provided
                top_k=body.get("top_k"),  # Let Google use its default if not provided
                max_output_tokens=body.get("max_tokens", 8192),  # Default max tokens
                stop_sequences=body.get(
                    "stop", None
                ),  # Pass stop sequences if provided
                # candidate_count=1 # Usually default, can be explicit
            )

            # Configure safety settings
            safety_settings = None
            if self.valves.USE_PERMISSIVE_SAFETY:
                safety_settings = {
                    # Block thresholds reference: https://ai.google.dev/docs/safety_setting_gemini
                    # BLOCK_NONE allows all content, even if potentially harmful. Use with caution.
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
                if DEBUG:
                    print(
                        "[pipe] Using permissive safety settings (BLOCK_NONE for all categories)."
                    )
            # else: You could allow passing specific safety settings via 'body' if needed.
            #    safety_settings = body.get("safety_settings", None) # Needs mapping from strings to enums

            if DEBUG:
                print("[pipe] Google API Request Details:")
                print(f"  Model: {model_id}")
                # print(f"  Contents: {contents}") # Content logged above more safely
                print(f"  Generation Config: {generation_config}")
                print(f"  Safety Settings: {safety_settings}")

            # --- Execute API Call Asynchronously ---
            try:
                # Define a synchronous helper for the executor
                def generate_content_sync(
                    client, contents, generation_config, safety_settings
                ):
                    # Note: stream=False for a single response object
                    if body.get("stream") == True:
                        return client.generate_content(
                            contents,
                            generation_config=generation_config,
                            safety_settings=safety_settings,
                            stream=True,
                        )
                    return client.generate_content(
                        contents,
                        generation_config=generation_config,
                        safety_settings=safety_settings,
                        stream=False,  # Get the full response at once
                    )

                # Run the synchronous SDK call in a thread pool executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,  # Use default executor
                    generate_content_sync,
                    client,
                    contents,
                    generation_config,
                    safety_settings,
                )
                if DEBUG:
                    if body.get("stream") == False:
                        # Safely log parts of the response
                        try:
                            log_resp = f"Candidates count: {len(response.candidates)}. "
                            if response.candidates:
                                part_count = len(response.candidates[0].content.parts)
                                log_resp += f"First candidate parts: {part_count}. Finish reason: {response.candidates[0].finish_reason.name}."
                                if part_count > 0:
                                    log_resp += f" First part text snippet: '{response.candidates[0].content.parts[0].text[:50]}...'"
                            if response.prompt_feedback:
                                log_resp += f" Prompt feedback block reason: {response.prompt_feedback.block_reason}"

                            print(
                                f"[pipe] Received response from Google API: {log_resp}"
                            )
                        except Exception as log_err:
                            print(f"[pipe] Error logging response details: {log_err}")

                # --- Process Response ---
                if body.get("stream") == True:
                    try:
                        for chunk in response:
                            if chunk.text:
                                yield chunk.text
                    except Exception as e:
                        if DEBUG:
                            print(f"[stream_generator] Error during streaming: {e}")
                        yield f"Error: {e}"
                    finally:
                        if DEBUG:
                            print("[stream_generator] Stream generator completed.")

                # Check for blocking first
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    reason = response.prompt_feedback.block_reason.name
                    await emit_error(f"Input prompt blocked due to: {reason}")
                    if DEBUG:
                        print(f"[pipe] Prompt blocked by API: {reason}")
                    return

                if not response.candidates:
                    # Check if response was blocked based on safety settings for the output
                    finish_reason = getattr(
                        response, "candidates[0].finish_reason", None
                    )  # Access safely
                    if finish_reason == genai.types.Candidate.FinishReason.SAFETY:
                        safety_ratings = getattr(
                            response, "candidates[0].safety_ratings", []
                        )
                        block_msg = "Response blocked due to safety settings."
                        if safety_ratings:
                            block_msg += " Categories: " + ", ".join(
                                [
                                    f"{r.category.name} ({r.probability.name})"
                                    for r in safety_ratings
                                ]
                            )
                        await emit_error(block_msg)
                        if DEBUG:
                            print(
                                f"[pipe] Response blocked by API safety settings: {safety_ratings}"
                            )
                    else:
                        await emit_error("Received no valid candidates from the model.")
                        if DEBUG:
                            print("[pipe] No candidates in response.")
                    return

                # Extract the main content (assuming non-streaming, single candidate)
                # Gemini might sometimes return empty parts, filter them out.
                valid_parts = [
                    part.text
                    for part in response.candidates[0].content.parts
                    if hasattr(part, "text") and part.text.strip()
                ]

                if valid_parts:
                    result_text = "\n".join(
                        valid_parts
                    )  # Join if multiple text parts exist
                    await emit_message(result_text)
                else:
                    # Check finish reason again for clues if parts are empty
                    finish_reason = response.candidates[0].finish_reason
                    if finish_reason == genai.types.Candidate.FinishReason.MAX_TOKENS:
                        await emit_error("Response stopped due to maximum token limit.")
                        if DEBUG:
                            print("[pipe] Response truncated due to max_tokens.")
                    elif finish_reason == genai.types.Candidate.FinishReason.STOP:
                        # This is normal if stop sequences were hit, but result might be empty
                        await emit_message(
                            ""
                        )  # Emit empty if stopped normally with no text
                        if DEBUG:
                            print(
                                "[pipe] Response finished due to stop sequence, but no text generated."
                            )
                    else:
                        await emit_error(
                            f"Received empty response content. Finish Reason: {finish_reason.name}"
                        )
                        if DEBUG:
                            print(
                                f"[pipe] Empty content received. Finish Reason: {finish_reason.name}"
                            )

            except genai.types.BlockedPromptException as bpe:
                if DEBUG:
                    print(f"[pipe] BlockedPromptException during API call: {bpe}")
                await emit_error(f"Input blocked by API safety settings: {bpe}")
            except genai.types.StopCandidateException as sce:
                if DEBUG:
                    print(f"[pipe] StopCandidateException during API call: {sce}")
                # This might happen if safety settings block the *output*
                await emit_error(
                    f"Output generation stopped, potentially due to safety settings: {sce}"
                )
            except Exception as e:
                # Catch potential API errors, network issues, etc.
                error_type = type(e).__name__
                if DEBUG:
                    print(f"[pipe] Error during API call ({error_type}): {e}")
                await emit_error(f"Error calling Google API: {error_type} - {e}")

        except Exception as e:
            # General catch-all for unexpected errors in the pipe logic
            if DEBUG:
                print(f"[pipe] Unexpected error in pipe execution: {e}")
            await emit_error(f"An unexpected error occurred: {e}")
        finally:
            if DEBUG:
                print("[pipe] Pipe execution finished.")
