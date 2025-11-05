import argparse
import json
from threading import Lock, Thread
from typing import List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DEFAULT_CKPT_PATH = "Qwen/Qwen2.5-7B-Instruct"


class HistoryTurn(BaseModel):
      user: str = Field(..., description="User message")
      assistant: str = Field(..., description="Assistant response")


class ChatRequest(BaseModel):
      prompt: str = Field(..., min_length=1, description="Prompt to send to the model")
      history: List[HistoryTurn] = Field(
            default_factory=list, description="Previous chat history"
      )
      stream: bool = Field(
            default=False, description="Return server-sent events instead of JSON"
      )


class ChatResponse(BaseModel):
      response: str
      history: List[HistoryTurn]


def _get_args():
      parser = argparse.ArgumentParser(description="Qwen2.5-Instruct API chat demo.")
      parser.add_argument(
            "-c",
            "--checkpoint-path",
            type=str,
            default=DEFAULT_CKPT_PATH,
            help="Checkpoint name or path, default to %(default)r",
      )
      parser.add_argument(
            "--cpu-only", action="store_true", help="Run inference on CPU only"
      )
      parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
      parser.add_argument("--port", type=int, default=8000, help="Server port")
      parser.add_argument(
            "--allow-origin",
            dest="allow_origins",
            action="append",
            default=[],
            help="Allowed CORS origins (pass multiple times). Defaults to '*' if omitted.",
      )
      parser.add_argument(
            "--no-cors",
            action="store_true",
            help="Disable CORS middleware entirely.",
      )
      return parser.parse_args()


def _load_model_tokenizer(args):
      tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint_path,
            resume_download=True,
      )

      if args.cpu_only:
            device_map = "cpu"
      else:
            device_map = "auto"

      model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_path,
            torch_dtype="auto",
            device_map=device_map,
            resume_download=True,
      ).eval()
      model.generation_config.max_new_tokens = 2048
      return model, tokenizer


def _chat_stream(model, tokenizer, query, history):
      conversation = []
      for query_h, response_h in history:
            conversation.append({"role": "user", "content": query_h})
            conversation.append({"role": "assistant", "content": response_h})
      conversation.append({"role": "user", "content": query})

      input_text = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
      )
      inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
      streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            skip_prompt=True,
            timeout=60.0,
            skip_special_tokens=True,
      )
      generation_kwargs = {**inputs, "streamer": streamer}

      thread = Thread(target=model.generate, kwargs=generation_kwargs)
      thread.start()
      try:
            for new_text in streamer:
                  yield new_text
      finally:
            thread.join()


def _gc():
      import gc

      gc.collect()
      if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _format_sse(event, data):
      return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(args, model, tokenizer):
      app = FastAPI(title="Qwen2.5-Instruct API Demo")
      generation_lock = Lock()

      if not args.no_cors:
            allow_origins = args.allow_origins or ["*"]
            app.add_middleware(
                  CORSMiddleware,
                  allow_origins=allow_origins,
                  allow_credentials=True,
                  allow_methods=["*"],
                  allow_headers=["*"],
            )

      @app.get("/healthz")
      async def healthz():
            return {"status": "ok"}

      @app.on_event("shutdown")
      def _on_shutdown():
            _gc()

      @app.post("/generate", response_model=ChatResponse)
      async def generate(request: ChatRequest):
            print(f"Received request: {request}")
            if not request.prompt.strip():
                  raise HTTPException(status_code=400, detail="prompt must not be empty")
            
            prompt = request.prompt
            history_pairs = [(turn.user, turn.assistant) for turn in request.history]

            print("start generation")
            if request.stream:
                  def event_stream():
                        response_text = ""
                        with generation_lock:
                              try:
                                    for token in _chat_stream(model, tokenizer, prompt, history_pairs):
                                          response_text += token
                                          yield _format_sse(
                                                "token",
                                                {"token": token, "done": False},
                                          ).encode("utf-8")
                              except Exception as exc:  # pragma: no cover
                                    yield _format_sse(
                                          "error",
                                          {"message": str(exc), "done": True},
                                    ).encode("utf-8")
                                    return

                              history_pairs.append((prompt, response_text))
                              payload = {
                                    "done": True,
                                    "response": response_text,
                                    "history": [
                                          {"user": user, "assistant": assistant}
                                          for user, assistant in history_pairs
                                    ],
                              }
                              yield _format_sse("result", payload).encode("utf-8")

                        return StreamingResponse(event_stream(), media_type="text/event-stream")

            with generation_lock:
                  response_text = ""
                  for token in _chat_stream(model, tokenizer, prompt, history_pairs):
                        response_text += token

            history_pairs.append((prompt, response_text))
            return ChatResponse(
                  response=response_text,
                  history=[
                        HistoryTurn(user=user, assistant=assistant)
                        for user, assistant in history_pairs
                  ],
            )

      return app


def main():
      args = _get_args()
      model, tokenizer = _load_model_tokenizer(args)
      app = create_app(args, model, tokenizer)
      uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
      main()