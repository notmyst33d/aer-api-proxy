# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Myst33d <myst33d@gmail.com>

import re
import httpx
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen3VLChatHandler

from .captcha_solver import CaptchaSolver
from .clip_classifier import CLIPClassifier
from .models import (
    ensure_models,
    CLASSIFIER_MODEL,
    CLASSIFIER_TOKENIZER,
    LLM_MODEL,
    LLM_MODEL_MMPROJ,
)

FAKE_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:146.0) Gecko/20100101 Firefox/146.0"
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global captcha_solver

    ensure_models()

    classifier = CLIPClassifier(
        model_path=CLASSIFIER_MODEL.path,
        tokenizer_path=CLASSIFIER_TOKENIZER.path,
    )
    chat_handler = Qwen3VLChatHandler(
        clip_model_path=LLM_MODEL_MMPROJ.path,
        verbose=False,
    )
    llm = Llama(
        model_path=LLM_MODEL.path,
        chat_handler=chat_handler,
        use_mmap=False,
        verbose=False,
    )

    captcha_solver = CaptchaSolver(classifier, llm)

    yield


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger("aer_api_proxy.captcha_solver").setLevel(logging.DEBUG)

app = FastAPI(lifespan=lifespan)

shared_cookies = {}


def build_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url="https://aliexpress.ru",
        headers={"User-Agent": FAKE_USER_AGENT},
        cookies=shared_cookies,
    )


@app.get("/aer-jsonapi/v1/bx/pdp/web/productData")
async def aer_jsonapi_v1_bx_pdp_web_product_data(request: Request):
    global shared_cookies

    async with build_client() as client:
        response = {"aer_api_proxy_message": "cannot access api"}
        attempts = 0
        while attempts < 3:
            client_response = await client.get(
                "/aer-jsonapi/v1/bx/pdp/web/productData",
                params=request.query_params,
            )

            # Handle captcha
            if client_response.headers.get("bxpunish"):
                try:
                    link = (
                        "https://"
                        + re.findall(r"(aliexpress\.ru.+?)\"", client_response.text)[0]
                    )
                    new_x5sec = await captcha_solver.run(
                        link, str(client_response.request.url)
                    )
                    if new_x5sec is None:
                        continue
                    client.cookies.update({"x5sec": new_x5sec})
                    shared_cookies.update({"x5sec": new_x5sec})
                except Exception as e:
                    logger.warning(f"failed to update x5sec: {e}")
                attempts += 1
                continue

            # Handle cookies
            if client_response.has_redirect_location:
                client_response = await client.get(client_response.headers["Location"])
                if xman_t := client.cookies.get("xman_t"):
                    shared_cookies.update({"xman_t": xman_t})
                if xman_f := client.cookies.get("xman_f"):
                    shared_cookies.update({"xman_f": xman_f})
                attempts += 1
                continue

            response = Response(
                content=client_response.content,
                status_code=client_response.status_code,
                media_type=client_response.headers["Content-Type"],
            )
            break
        return response
