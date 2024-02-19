import re
import enum
from typing import Tuple, Union

from .utils import (
    ACTION_PARSER_MAP,
    PROMPT_MAP,
    CompletionGPT,
    ChatGPT,
    ChatAnthropic,
    PalmChat,
    PalmCompletion,
    HFChat,
)


class ModelType(enum.Enum):
    Anthropic = enum.auto()
    OpenAI = enum.auto()


class DialogueController:
    def __init__(
        self,
        model_type: ModelType,
        context_window_size: int,
        dialogue_limit: int = 40,
        observation_limit: Union[int, None] = None,
        num_char_by_token: int = 3,
    ):
        self.model_type = model_type
        self.context_window_size = context_window_size
        self.dialogue_limit = dialogue_limit
        self.num_char_by_token = num_char_by_token
        self._set_observation_limit(observation_limit)
        self.dialogue = []

    def __str__(self):
        return (
            f"Model Type: {self.model_type}\n"
            f"Context Window Size: {self.context_window_size}\n"
            f"Dialogue Limit: {self.dialogue_limit}\n"
            f"Observation Limit: {self.observation_limit}"
        )

    def _set_observation_limit(self, observation_limit):
        """Sets the observation to 3 turn if not set before."""
        if observation_limit is not None:
            self.observation_limit = observation_limit
        all_size = self.context_window_size * self.num_char_by_token
        turn_size = all_size // self.dialogue_limit
        self.observation_limit = turn_size * 3

    def reset(self):
        self.dialogue = []

    def __len__(self):
        return len(self.dialogue)

    def append(self, turn):
        observation = turn["content"]

        if observation is not None and len(observation) > self.observation_limit:
            observation = observation[: self.observation_limit] + "... [Truncated]"

        turn["content"] = observation
        self.dialogue.append(turn)
        self._limit_turn()

    def _limit_turn(self):
        # Only keep {self.dialogue_limit} most recent messages
        num_head_dialogue = 1 if self.model_type == ModelType.Anthropic else 2
        if self.dialogue_limit and len(self) - num_head_dialogue > self.dialogue_limit:
            self.dialogue = (
                self.dialogue[:num_head_dialogue]
                + self.dialogue[-self.dialogue_limit :]
            )
        # TODO: limit according to context length

    def __getitem__(self, index):
        return self.dialogue[index]


def initialize_policy(policy_type, model, **kwargs) -> "BasePolicy":
    if model == "claude":
        policy = ChatAnthropicPolicy(model=model, **kwargs)
    elif policy_type == "chat":
        policy = ChatGPTPolicy(model=model, **kwargs)
    elif policy_type == "complete":
        policy = CompletionGPTPolicy(model=model, **kwargs)
    else:
        raise ValueError(f"Policy {policy_type!r} not recognized")
    return policy


class BasePolicy:
    def __init__(self):
        pass

    def forward(query, observation):
        raise NotImplementedError


class HumanPolicy(BasePolicy):
    def __init__(self):
        super().__init__()

    def forward(self, query, observation):
        action = input("Human Action > ")
        return action


class CompletionGPTPolicy(BasePolicy):
    def __init__(
        self,
        language: str,
        setting: str,
        template: str,
        dialogue_limit: int = None,
        model: str = "text-davinci-003",
        response_limit: int = 500,
    ):
        super().__init__()
        self.language = language.upper()
        self.setting = setting
        self.template = PROMPT_MAP[template](self.language, self.setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.response_limit = response_limit
        self.model = model
        self.handicap = ""
        self.dialogue_limit = dialogue_limit

    def reset(self):
        self.prompt = None
        self.dialogue = {}

    def add_to_dialogue(self, handicap: str):
        self.handicap = handicap

    def forward(self, query, observation, reward) -> Tuple[str, bool]:
        if self.prompt is None:
            # First Turn
            prompt = self.prompt = (
                self.template.get_init_msg()
                + self.handicap
                + self.template.get_query_msg(query)
            )
        else:
            # Limit observation size due to context window thresholds for API call
            if isinstance(observation, str) and len(observation) > self.response_limit:
                observation = observation[: self.response_limit]
            elif isinstance(observation, list) and len(observation) > 50:
                observation = observation[:50]
            if (
                isinstance(observation, str)
                and observation == ""
                or isinstance(observation, list)
                and len(observation) == 0
            ):
                observation = "No output"
            self.dialogue["reward"] = reward
            self.dialogue["observations"] = observation
            # N-th Turn
            prompt = self.prompt + self.template.get_obs_msg(self.dialogue)

        # Retrieve Completion GPT
        actions = CompletionGPT(prompt, model=self.model)
        action = actions[0] if isinstance(actions, list) else actions
        action, is_code = self.action_parser(action)
        self.dialogue["actions"] = action
        return action, is_code


class ChatGPTPolicy(BasePolicy):
    def __init__(
        self,
        language: str,
        setting: str,
        template: str,
        model: str = "gpt-3.5-turbo",
        **dialog_kwargs,
    ):
        super().__init__()
        self.language = language.upper()
        self.dialogue_controller = DialogueController(
            model_type=ModelType.OpenAI,
            context_window_size=16_000 if model.startswith("gpt-3.5") else 128_000,
            **dialog_kwargs,
        )
        self.setting = setting
        self.template = PROMPT_MAP[template](self.language, self.setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.model = model

    def reset(self):
        self.dialogue_controller.reset()
        self.dialogue_controller.append(
            {"role": "system", "content": self.template.get_init_msg()}
        )

    def add_to_dialogue(self, handicap: str):
        self.dialogue_controller.append({"role": "system", "content": handicap})

    def forward(self, query, observation, reward) -> Tuple[str, bool]:
        # Append response to dialogue
        if self.dialogue_controller[-1]["role"] == "system":
            # First Turn
            self.dialogue_controller.append(
                {"role": "user", "content": self.template.get_query_msg(query)}
            )
        else:
            self.dialogue_controller.append(
                {
                    "role": "user",
                    "content": self.template.get_obs_msg(observation, reward),
                }
            )

        actions = ChatGPT(self.dialogue_controller.dialogue, model=self.model)
        action = actions[0] if isinstance(actions, list) else actions
        self.dialogue_controller.append({"role": "assistant", "content": action})
        return action, True


class ChatAnthropicPolicy(BasePolicy):
    def __init__(
        self,
        language: str,
        setting: str,
        template: str,
        model: str = "",
        max_tokens=512,
        temperature=0,
        top_p=1,
        **dialog_kwargs,
    ):
        super().__init__()
        self.language = language.upper()
        self.dialogue_controller = DialogueController(
            model_type=ModelType.Anthropic,
            context_window_size=200_000,
            **dialog_kwargs,
        )
        self.template = PROMPT_MAP[template](self.language, setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def reset(self):
        self.dialogue_controller.reset()

    def forward(
        self, query: str, observation: Union[str, None], reward: Union[float, None]
    ) -> Tuple[str, bool]:
        if not len(self.dialogue_controller):
            self.dialogue_controller.append(
                {"role": "user", "content": self.template.get_query_msg(query)}
            )
        else:
            self.dialogue_controller.append(
                {
                    "role": "user",
                    "content": self.template.get_obs_msg(observation, reward),
                }
            )

        action = ChatAnthropic(
            self.dialogue_controller.dialogue,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            system=self.template.get_init_msg(),
        )
        self.dialogue_controller.append({"role": "assistant", "content": action})
        return action, True


class PalmChatPolicy(BasePolicy):
    def __init__(
        self,
        language: str,
        setting: str,
        template: str,
        dialogue_limit: int = None,
        model: str = "models/chat-bison-001",
        response_limit: int = 1000,
    ):
        super().__init__()
        self.language = language.upper()
        self.setting = setting
        self.template = PROMPT_MAP[template](self.language, self.setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.response_limit = response_limit
        self.model = model
        self.handicap = ""

    def reset(self):
        self.chatbot = PalmChat(self.model)
        self.dialogue = []

    def add_to_dialogue(self, handicap: str):
        self.handicap = handicap

    def forward(self, query, observation, reward) -> Tuple[str, bool]:
        if len(self.dialogue) == 0:
            # First Turn
            self.dialogue = [
                {
                    "author": "0",
                    "content": self.template.get_init_msg()
                    + self.handicap
                    + self.template.get_query_msg(query),
                }
            ]
            self.chatbot.init_chat(init_message=self.dialogue)
        else:
            # Limit observation size due to context window thresholds for API call
            if isinstance(observation, str) and len(observation) > self.response_limit:
                observation = observation[: self.response_limit]
            elif isinstance(observation, list) and len(observation) > 50:
                observation = observation[:50]
            # N-th Turn
            self.chatbot.reply(self.template.get_obs_msg(observation, reward))

        # Retrieve Action from PalmChat
        action = self.chatbot.get_response()
        action, is_code = self.action_parser(action)
        return action, is_code


class PalmCompletionPolicy(BasePolicy):
    def __init__(
        self,
        language: str,
        setting: str,
        template: str,
        dialogue_limit: int = None,
        model: str = "models/text-bison-001",
        response_limit: int = 1000,
    ):
        super().__init__()
        self.language = language.upper()
        self.setting = setting
        self.template = PROMPT_MAP[template](self.language, self.setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.response_limit = response_limit
        self.model = model
        self.handicap = ""

    def reset(self):
        self.prompt = None
        self.dialogue = {}

    def add_to_dialogue(self, handicap: str):
        self.handicap = handicap

    def forward(self, query, observation, reward) -> Tuple[str, bool]:
        if self.prompt is None:
            # First Turn
            prompt = self.prompt = (
                self.template.get_init_msg()
                + self.handicap
                + self.template.get_query_msg(query)
            )
        else:
            # Limit observation size due to context window thresholds for API call
            if isinstance(observation, str) and len(observation) > self.response_limit:
                observation = observation[: self.response_limit]
            elif isinstance(observation, list) and len(observation) > 50:
                observation = observation[:50]
            if (
                isinstance(observation, str)
                and observation == ""
                or isinstance(observation, list)
                and len(observation) == 0
            ):
                observation = "No output"
            self.dialogue["reward"] = reward
            self.dialogue["observations"] = observation
            # N-th Turn
            prompt = self.prompt + self.template.get_obs_msg(self.dialogue)

        # Retrieve Action from PalmChat
        actions = PalmCompletion(prompt, model=self.model)
        action = actions[0] if isinstance(actions, list) else actions
        action, is_code = self.action_parser(action)
        self.dialogue["actions"] = action
        return action, is_code


class HFChatPolicy(BasePolicy):
    def __init__(
        self,
        language: str,
        setting: str,
        template: str,
        dialogue_limit: int = None,
        model: str = "gpt-3.5-turbo",
        response_limit: int = 1000,
    ):
        super().__init__()
        self.language = language.upper()
        self.setting = setting
        self.dialogue_limit = dialogue_limit
        self.template = PROMPT_MAP[template](self.language, self.setting)
        self.action_parser = ACTION_PARSER_MAP[language]
        self.model = model
        self.response_limit = response_limit

    def reset(self):
        self.dialogue = []

    def add_to_dialogue(self, handicap: str):
        self.dialogue.append({"role": "system", "content": handicap})

    def forward(self, query, observation, reward) -> Tuple[str, bool]:
        # Append response to dialogue
        if len(self.dialogue) == 0:
            # First Turn
            self.dialogue = [
                {"role": "<|system|>", "content": "You are an expert in Bash."}
            ]
            self.dialogue.append(
                {
                    "role": "<|user|>",
                    "content": self.template.get_init_msg()
                    + self.template.get_query_msg(query),
                }
            )
        else:
            # Limit observation size due to context window thresholds for API call
            if isinstance(observation, str) and len(observation) > self.response_limit:
                observation = observation[: self.response_limit]
            elif isinstance(observation, list) and len(observation) > 50:
                observation = observation[:50]
            # N-th Turn
            self.dialogue.append(
                {
                    "role": "<|user|>",
                    "content": self.template.get_obs_msg(observation, reward),
                }
            )
            # Only keep {self.dialogue_limit} most recent messages
            if self.dialogue_limit and len(self.dialogue) - 2 > self.dialogue_limit:
                self.dialogue = (
                    self.dialogue[:2] + self.dialogue[-self.dialogue_limit :]
                )

        chat = ""
        for d in self.dialogue:
            chat += f"{d['role']} {d['content']} <|end|>\n"
        actions = HFChat(chat + "<|assistant|>")
        action = actions[0] if isinstance(actions, list) else actions
        action, is_code = self.action_parser(action)
        self.dialogue.append({"role": "<|assistant|>", "content": action})
        return action, is_code
