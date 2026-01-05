# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Myst33d <myst33d@gmail.com>

import random
import base64
import numpy
import logging
from typing import Iterator, Optional, Tuple, Union, cast
from llama_cpp import Llama
from io import BytesIO
from PIL import Image
from playwright.async_api import Locator, async_playwright, Page, ElementHandle

from .clip_classifier import CLIPClassifier


class CaptchaSolver:
    def __init__(self, classifier: CLIPClassifier, llm: Llama):
        self._logger = logging.getLogger(__name__)
        self._classifier = classifier
        self._llm = llm

    async def run(self, referer: str, link: str) -> Optional[str]:
        async with async_playwright() as p:
            self._logger.debug("loading captcha page")

            browser = await p.chromium.launch(
                args=["--disable-blink-features=AutomationControlled"]
            )
            context = await browser.new_context(has_touch=True)

            page = await context.new_page()
            await page.goto(link, referer=referer)
            await page.wait_for_selector('#click-grid-0[style*="opacity: 1"]')

            for i in range(3):
                self._logger.debug(f"trying to solve captcha, attempt {i + 1}")
                if await self._try_solve(page):
                    self._logger.debug("successfully solved captcha")
                    x5sec = next(
                        filter(
                            lambda x: x["name"] == "x5sec",  # type: ignore
                            await page.context.cookies(),
                        )
                    )["value"]
                    await browser.close()
                    return x5sec
                await page.wait_for_timeout(4000)
            await browser.close()

        return None

    async def _try_solve(self, page: Page) -> bool:
        self._logger.debug("preparing image question for llm")
        prompt_canvas = page.locator("#click-question-canvas")
        filter_images = []
        for _ in range(10):
            im = await _canvas_to_image(prompt_canvas)
            filter_images.append(im)

        prompt_im = _filter_prompt_image(filter_images)
        object_name = await self._llm_ocr(prompt_im)
        self._logger.debug(f'llm ocr\'d the image question to "{object_name}"')

        while True:
            tiles = []
            for tile in await page.query_selector_all(
                "#click-captcha-question-container .grid"
            ):
                im = await _canvas_to_image(
                    cast(ElementHandle, await tile.query_selector("canvas"))
                )
                tiles.append((tile, im))
            random.shuffle(tiles)

            match_atleast_one = False
            for tile, im in tiles:
                tile_id = await (await tile.query_selector("canvas")).get_attribute(
                    "id"
                )
                is_match, prob = await self._classifier_is_match(object_name, im)
                match_atleast_one |= is_match
                if is_match:
                    self._logger.debug(
                        f'classifier thinks tile {tile_id} matches "{object_name}" (probability: {prob})'
                    )
                    await _realistic_click(page, tile)
                    await page.wait_for_selector(
                        f'#{tile_id}[style*="opacity: 1"]',
                        timeout=3000,
                    )
                    break
            if not match_atleast_one:
                self._logger.debug("no more matching objects")
                await _realistic_click(page, page.locator("#click-submit"))
                async with page.expect_response(
                    lambda r: r.url.find("gridClickVerify") != -1
                ) as response_info:
                    response = await response_info.value
                    data = await response.json()
                self._logger.debug(f"code: {data['code']}")
                if data["code"] == 0:
                    return True
                else:
                    return False

    async def _llm_ocr(self, image: Image.Image) -> str:
        response = self._llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "Extract the text from the image. Do not add anything else.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": _image_to_data_url(image),
                        }
                    ],
                },
            ],
            temperature=0,
        )
        if isinstance(response, Iterator):
            raise Exception("response is an Iterator instance")
        if not isinstance(response["choices"][0]["message"]["content"], str):
            raise Exception("content is empty")
        return response["choices"][0]["message"]["content"]

    async def _classifier_is_match(
        self, object_name: str, image: Image.Image
    ) -> Tuple[bool, float]:
        probs = self._classifier.run(image, [object_name, "other"]).tolist()[0]
        return (probs[0] >= 0.98, probs[0])


async def _realistic_click(page: Page, el: Union[ElementHandle, Locator]):
    bbox = await el.bounding_box()
    if bbox is None:
        raise Exception("bbox is None")
    await page.touchscreen.tap(
        bbox["x"] + bbox["width"] / 2,
        bbox["y"] + bbox["height"] / 2,
    )


def _filter_prompt_image(images: list[Image.Image]) -> Image.Image:
    alpha_mult = 1.0 / len(images)
    comp = numpy.array(images[0]) * alpha_mult
    for i in range(1, len(images)):
        comp = comp + (numpy.array(images[i]) * alpha_mult)
    prompt_im = (
        numpy.array(Image.fromarray(numpy.uint8(comp)).convert("L"), dtype="float")
        / 255.0
    )
    avg_l = prompt_im.mean()
    for i, _ in numpy.ndenumerate(prompt_im):
        p00 = prompt_im[i[0]][i[1]]
        p01 = avg_l
        if i[1] + 1 < prompt_im.shape[1]:
            p01 = prompt_im[i[0]][i[1] + 1]
        p10 = avg_l
        if i[0] + 1 < prompt_im.shape[0]:
            p10 = prompt_im[i[0] + 1][i[1]]
        p11 = avg_l
        if i[0] + 1 < prompt_im.shape[0] and i[1] + 1 < prompt_im.shape[1]:
            p11 = prompt_im[i[0] + 1][i[1] + 1]
        avg_l_block = (p00 + p01 + p10 + p11) / 4
        prompt_im[i[0]][i[1]] = -(avg_l_block - avg_l + 0.015)
    return Image.fromarray(numpy.uint8(prompt_im * 255.0))


def _image_to_data_url(image: Image.Image) -> str:
    b = BytesIO()
    image.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


async def _canvas_to_image(canvas: ElementHandle | Locator) -> Image.Image:
    return Image.open(
        BytesIO(base64.b64decode((await canvas.evaluate("el => el.toDataURL()"))[22:]))
    )
