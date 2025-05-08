import copy
import re
import traceback
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable, Literal, Type, TypeVar

import altair as alt
import commentjson
import jsonschema
import pandas as pd
import pydantic
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, FunctionMessage, HumanMessage, SystemMessage
from plotly.graph_objs import Figure


import os, sys
print (os.getcwd())
module_path = "../.."
sys.path.append(os.path.abspath(module_path))

from utils.custom_chat2plot.dataset_description import description
from utils.custom_chat2plot.dictionary_helper import delete_null_field
from utils.custom_chat2plot.prompt import (
    JSON_TAG,
    error_correction_prompt,
    explanation_prompt,
    system_prompt,
)
from utils.custom_chat2plot.render import draw_altair, draw_plotly
from utils.custom_chat2plot.schema import PlotConfig, ResponseType, get_schema_of_chart_config

_logger = getLogger(__name__)

T = TypeVar("T", bound=pydantic.BaseModel)
ModelDeserializer = Callable[[dict[str, Any]], T]

# These errors are caught within the application.
# Other errors (e.g. openai.error.RateLimitError) are propagated to user code.
_APPLICATION_ERRORS = (
    pydantic.ValidationError,
    jsonschema.ValidationError,
    ValueError,
    KeyError,
    AssertionError,
)


@dataclass(frozen=True)
class Plot:
    figure: alt.Chart | Figure | None
    config: PlotConfig | dict[str, Any] | pydantic.BaseModel | None
    response_type: ResponseType
    explanation: str
    conversation_history: list[BaseMessage] | None


class ChatSession:
    """chat with conversasion history"""

    def __init__(
        self,
        chat: BaseChatModel,
        df: pd.DataFrame,
        system_prompt_template: str,
        user_prompt_template: str,
        description_strategy: str = "head",
        functions: list[dict[str, Any]] | None = None,
    ):
        self._system_prompt_template = system_prompt_template
        self._user_prompt_template = user_prompt_template
        self._chat = chat
        self._conversation_history: list[BaseMessage] = [
            SystemMessage(
                content=system_prompt_template.format(
                    dataset=description(df, description_strategy)
                )
            )
        ]
        self._functions = functions

    @property
    def history(self) -> list[BaseMessage]:
        return copy.deepcopy(self._conversation_history)

    def query(self, q: str, raw: bool = False) -> BaseMessage:
        prompt = q if raw else self._user_prompt_template.format(text=q)
        response = self._query(prompt)
        return response

    def _query(self, prompt: str) -> BaseMessage:
        self._conversation_history.append(HumanMessage(content=prompt))
        kwargs = {}
        if self._functions:
            kwargs["functions"] = self._functions
        response = self._chat(self._conversation_history, **kwargs)  # type: ignore
        self._conversation_history.append(response)

        if response.additional_kwargs.get("function_call"):
            name = response.additional_kwargs["function_call"]["name"]
            arguments = response.additional_kwargs["function_call"]["arguments"]
            self._conversation_history.append(
                FunctionMessage(name=name, content=arguments)
            )

        return response

    def last_response(self) -> str:
        return self._conversation_history[-1].content


class Chat2PlotBase:
    @property
    def session(self) -> ChatSession:
        raise NotImplementedError()

    @property
    def function_call(self) -> bool:
        return False

    def query(self, q: str, config_only: bool = False, show_plot: bool = False) -> Plot:
        raise NotImplementedError()

    def __call__(
        self, q: str, config_only: bool = False, show_plot: bool = False
    ) -> Plot:
        return self.query(q, config_only, show_plot)


class Chat2Plot(Chat2PlotBase):
    def __init__(
        self,
        df: pd.DataFrame,
        chart_schema: Literal["simple"] | Type[pydantic.BaseModel],
        *,
        chat: BaseChatModel | None = None,
        function_call: bool | Literal["auto"] = False,
        language: str | None = None,
        description_strategy: str = "head",
        verbose: bool = False,
        custom_deserializer: ModelDeserializer | None = None,
    ):
        self._target_schema: Type[pydantic.BaseModel] = (
            PlotConfig if chart_schema == "simple" else chart_schema  # type: ignore
        )

        chat_model = _get_or_default_chat_model(chat)

        self._function_call = (
            _has_function_call_capability(chat_model)
            if function_call == "auto"
            else function_call
        )

        self._session = ChatSession(
            chat_model,
            df,
            system_prompt("simple", self._function_call, language, self._target_schema),
            "<{text}>",
            description_strategy,
            functions=[
                get_schema_of_chart_config(self._target_schema, as_function=True)
            ]
            if self._function_call
            else None,
        )
        self._df = df
        self._verbose = verbose
        self._custom_deserializer = custom_deserializer
        self._language = language

    @property
    def session(self) -> ChatSession:
        return self._session

    @property
    def function_call(self) -> bool:
        return self._function_call

    def query(self, q: str, config_only: bool = False, show_plot: bool = False) -> Plot:
        raw_response = self._session.query(q)
        
        
        
        
        
        
        
        

        try:
            if self._verbose:
                _logger.info(f"request: {q}")
                _logger.info(f"first response: {raw_response}")
            return self._parse_response(q, raw_response, config_only, show_plot)
        except _APPLICATION_ERRORS as e:
            if self._verbose:
                _logger.warning(traceback.format_exc())
            msg = e.message if isinstance(e, jsonschema.ValidationError) else str(e)
            error_correction = error_correction_prompt(self._function_call).format(
                error_message=msg,
            )
            corrected_response = self._session.query(error_correction)
            if self._verbose:
                _logger.info(f"retry response: {corrected_response}")

            try:
                return self._parse_response(
                    q, corrected_response, config_only, show_plot
                )
            except _APPLICATION_ERRORS as e:
                if self._verbose:
                    _logger.warning(e)
                    _logger.warning(traceback.format_exc())
                return Plot(
                    None,
                    None,
                    ResponseType.FAILED_TO_RENDER,
                    "",
                    self._session.history,
                )

    def __call__(
        self, q: str, config_only: bool = False, show_plot: bool = False
    ) -> Plot:
        return self.query(q, config_only, show_plot)

    def _parse_response(
        self, q: str, response: BaseMessage, config_only: bool, show_plot: bool
    ) -> Plot:
        if self._function_call:
            if not response.additional_kwargs.get("function_call"):
                raise ValueError("Function should be called")
            function_call = response.additional_kwargs["function_call"]
            json_data = commentjson.loads(function_call["arguments"])

            explanation = self._session.query(
                explanation_prompt(self._language, q), raw=True
            ).content
        else:
            explanation, json_data = parse_json(response.content)

        try:
            if self._custom_deserializer:
                config = self._custom_deserializer(json_data)
            else:
                config = self._target_schema.parse_obj(json_data)
        except _APPLICATION_ERRORS:
            _logger.warning(traceback.format_exc())
            # To reduce the number of failure cases as much as possible,
            # only check against the json schema when instantiation fails.
            jsonschema.validate(json_data, self._target_schema.schema())
            raise

        if self._verbose:
            _logger.info(config)

        if config_only or not isinstance(config, PlotConfig):
            return Plot(
                None, config, ResponseType.SUCCESS, explanation, self._session.history
            )

        figure = draw_plotly(self._df, config, show_plot)
        return Plot(
            figure, config, ResponseType.SUCCESS, explanation, self._session.history
        )


class Chat2Vega(Chat2PlotBase):
    def __init__(
        self,
        df: pd.DataFrame,
        chat: BaseChatModel | None = None,
        language: str | None = None,
        description_strategy: str = "head",
        verbose: bool = False,
    ):
        self._session = ChatSession(
            _get_or_default_chat_model(chat),
            df,
            system_prompt("vega", False, language, None),
            "<{text}>",
            description_strategy,
        )
        self._df = df
        self._verbose = verbose

    @property
    def session(self) -> ChatSession:
        return self._session

    def query(self, q: str, config_only: bool = False, show_plot: bool = False) -> Plot:
        res = self._session.query(q)

        try:
            explanation, config = parse_json(res.content)
            if "data" in config:
                del config["data"]
            if self._verbose:
                _logger.info(config)
        except _APPLICATION_ERRORS:
            _logger.warning(f"failed to parse LLM response: {res}")
            _logger.warning(traceback.format_exc())
            return Plot(
                None, None, ResponseType.UNKNOWN, res.content, self._session.history
            )

        if config_only:
            return Plot(
                None, config, ResponseType.SUCCESS, explanation, self._session.history
            )

        try:
            plot = draw_altair(self._df, config, show_plot)
            return Plot(
                plot, config, ResponseType.SUCCESS, explanation, self._session.history
            )
        except _APPLICATION_ERRORS:
            _logger.warning(traceback.format_exc())
            return Plot(
                None,
                config,
                ResponseType.FAILED_TO_RENDER,
                explanation,
                self._session.history,
            )

    def __call__(
        self, q: str, config_only: bool = False, show_plot: bool = False
    ) -> Plot:
        return self.query(q, config_only, show_plot)


def chat2plot(
    df: pd.DataFrame,
    schema_definition: Literal["simple", "vega"] | Type[pydantic.BaseModel] = "simple",
    chat: BaseChatModel | None = None,
    function_call: bool | Literal["auto"] = "auto",
    language: str | None = None,
    description_strategy: str = "head",
    custom_deserializer: ModelDeserializer | None = None,
    verbose: bool = False,
) -> Chat2PlotBase:
    """Create Chat2Plot instance.

    Args:
        df: Data source for visualization.
        schema_definition: Type of json format; "vega" for vega-lite compliant json, "simple" for chat2plot built-in
              data structure. If you want a custom schema definition, pass a type inheriting from pydantic.BaseModel
              as your own chart setting.
        chat: The chat instance for interaction with LLMs.
              If omitted, `ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")` will be used.
        function_call:
        language: Language of explanations. If not specified, it will be automatically inferred from user prompts.
        description_strategy: Type of how the information in the dataset is embedded in the prompt.
              Defaults to "head" which embeds the contents of df.head(5) in the prompt.
              "dtypes" sends only columns and types to LLMs and does not send the contents of the dataset,
              which allows for privacy but may reduce accuracy.
        custom_deserializer: A custom function to convert the json returned by the LLM into a object.
        verbose: If `True`, chat2plot will output logs.

    Returns:
        Chat instance.
    """

    if schema_definition == "simple":
        return Chat2Plot(
            df,
            "simple",
            chat=chat,
            language=language,
            description_strategy=description_strategy,
            verbose=verbose,
            custom_deserializer=custom_deserializer,
            function_call=function_call,
        )
    if schema_definition == "vega":
        return Chat2Vega(df, chat, language, description_strategy, verbose)
    elif issubclass(schema_definition, pydantic.BaseModel):
        return Chat2Plot(
            df,
            schema_definition,
            chat=chat,
            language=language,
            description_strategy=description_strategy,
            verbose=verbose,
            custom_deserializer=custom_deserializer,
            function_call=function_call,
        )
    else:
        raise ValueError(
            f"schema_definition should be one of [simple, vega] or pydantic.BaseClass (given: {schema_definition})"
        )


def _extract_tag_content(s: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*)</{tag}>", s, re.MULTILINE | re.DOTALL)
    if m:
        return m.group(1)
    else:
        m = re.search(rf"<{tag}>(.*)<{tag}>", s, re.MULTILINE | re.DOTALL)
        if m:
            return m.group(1)
    return ""


def parse_json(content: str) -> tuple[str, dict[str, Any]]:
    """parse json and split contents by pre-defined tags"""
    json_part = _extract_tag_content(content, "json")  # type: ignore
    if not json_part:
        raise ValueError(f"failed to find {JSON_TAG[0]} and {JSON_TAG[1]} tags")

    explanation_part = _extract_tag_content(content, "explain")
    if not explanation_part:
        explanation_part = _extract_tag_content(content, "explanation")

    # LLM rarely generates JSON with comments, so use the commentjson package instead of json
    return explanation_part.strip(), delete_null_field(commentjson.loads(json_part))


def _get_or_default_chat_model(chat: BaseChatModel | None) -> BaseChatModel:
    if chat is None:
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0613")  # type: ignore
    return chat


def _has_function_call_capability(chat: BaseChatModel) -> bool:
    if not isinstance(chat, ChatOpenAI):
        return False
    return any(
        chat.model_name.startswith(prefix)
        for prefix in ["gpt-4-0613", "gpt-3.5-turbo-0613"]
    )
